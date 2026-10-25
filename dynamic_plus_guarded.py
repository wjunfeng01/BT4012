
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from dynamic_adaptive import (
    SEED,
    VALIDATION_WINDOWS,
    ensemble_probabilities,
    load_data,
    make_submission,
    model_for,
    model_predictions,
    model_predictions_saved,
    safe_logit,
)


def recency_predictions(data, cutoff, x_eval):
    rows = data["known"] & (data["steps"] <= cutoff)
    weights = 0.35 + 0.65 * np.exp(-0.085 * (cutoff - data["steps"][rows]))
    standard = model_for("LightGBM", "known", seed=SEED)
    recent = model_for("LightGBM", "known", seed=SEED + 19)
    standard.fit(data["x_train"][rows], data["target"][rows])
    recent.fit(data["x_train"][rows], data["target"][rows], sample_weight=weights)
    return standard.predict_proba(x_eval)[:, 1], recent.predict_proba(x_eval)[:, 1]




def recency_predictions_saved(model_dir, evaluation_features):
    from lightgbm import Booster

    results = []
    for name in ("standard", "recent"):
        model = Booster(model_file=str(model_dir / f"specialist_{name}.txt"))
        results.append(model.predict(evaluation_features, num_threads=4))
    return tuple(results)

def guarded_probabilities(dynamic, standard, recent):
    difference = np.clip(safe_logit(recent) - safe_logit(standard), -1.5, 1.5)
    return expit(safe_logit(dynamic) + 0.035 * difference)


def validate(data):
    rows = []
    for cutoff, start, end in VALIDATION_WINDOWS:
        mask = data["known"] & (data["steps"] >= start) & (data["steps"] <= end)
        scores = model_predictions(data, cutoff, data["x_train"][mask])
        fixed, dynamic = ensemble_probabilities(scores)
        standard, recent = recency_predictions(data, cutoff, data["x_train"][mask])
        guarded = guarded_probabilities(dynamic, standard, recent)
        y = data["target"][mask]
        rows.append({
            "train_through": cutoff,
            "validation_steps": f"{start}-{end}",
            "n": int(mask.sum()),
            "illicit_n": int(y.sum()),
            "fixed_auc": roc_auc_score(y, fixed),
            "dynamic_auc": roc_auc_score(y, dynamic),
            "guarded_auc": roc_auc_score(y, guarded),
            "guarded_average_precision": average_precision_score(y, guarded),
        })
    return pd.DataFrame(rows)


def generate_both(data, root):
    cutoff = int(data["steps"].max())
    model_dir = root / "models"
    if os.environ.get("RETRAIN_FROM_SCRATCH", "0") == "1":
        base_scores = model_predictions(data, cutoff, data["x_test"])
        standard, recent = recency_predictions(data, cutoff, data["x_test"])
    else:
        if not model_dir.is_dir():
            raise FileNotFoundError("Saved models are missing. Restore models/ or set RETRAIN_FROM_SCRATCH=1 to train again.")
        base_scores = model_predictions_saved(model_dir, data["x_test"])
        standard, recent = recency_predictions_saved(model_dir, data["x_test"])
        print("Using the saved fitted models for final inference")
    _, dynamic = ensemble_probabilities(base_scores)
    dynamic_csv = make_submission(data, dynamic, root / "dynamic_adaptive_submission_7.csv")
    guarded = guarded_probabilities(dynamic, standard, recent)
    guarded_csv = make_submission(data, guarded, root / "dynamic_plus_guarded_8.csv")
    return dynamic_csv, guarded_csv


def main():
    root = Path(__file__).resolve().parent
    data = load_data(os.environ.get("BITCOIN_DATA_DIR", root / "data"))
    if os.environ.get("SKIP_VALIDATION", "0") != "1":
        result = validate(data)
        result.to_csv(root / "validation_results.csv", index=False)
        print(result.round(6).to_string(index=False))
    generate_both(data, root)


if __name__ == "__main__":
    main()
