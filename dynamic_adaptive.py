
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


SEED = 2026
N_JOBS = 4
FEATURES = [f"feat_{i}" for i in range(1, 166)]
ALGORITHMS = ("LightGBM", "XGBoost", "CatBoost")
ALGORITHM_WEIGHTS = (0.40, 0.20, 0.40)
VALIDATION_WINDOWS = ((24, 25, 28), (28, 29, 30), (30, 31, 35))


def read_table(data_dir, filename):
    path = data_dir / filename
    if path.is_file():
        return pd.read_csv(path)
    archives = [data_dir / "bt-4012-competition-2026.zip", *sorted(data_dir.glob("*.zip"))]
    for archive_path in archives:
        if not archive_path.is_file():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            matches = [p for p in archive.namelist() if Path(p).name == filename]
            if len(matches) == 1:
                with archive.open(matches[0]) as stream:
                    return pd.read_csv(stream)
    raise FileNotFoundError(f"Missing {filename} in {data_dir}")


def load_data(data_dir):
    data_dir = Path(data_dir)
    train = read_table(data_dir, "train.csv")
    test = read_table(data_dir, "test.csv")
    template = read_table(data_dir, "sample_submission.csv")

    x_train = train[FEATURES].to_numpy(dtype=np.float32)
    x_test = test[FEATURES].to_numpy(dtype=np.float32)
    steps = train["time_step"].to_numpy()
    labels = pd.to_numeric(train["label"], errors="coerce").to_numpy()
    known = np.isfinite(labels)
    target = (labels == 1).astype(np.int8)

    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Feature matrix contains missing or infinite values")
    if not template["index"].equals(test["index"]):
        raise ValueError("Sample submission and test indices are not aligned")

    return {
        "train": train,
        "test": test,
        "template": template,
        "x_train": x_train,
        "x_test": x_test,
        "steps": steps,
        "target": target,
        "known": known,
    }


def model_for(algorithm, objective, seed=SEED):
    if algorithm == "LightGBM":
        return LGBMClassifier(
            n_estimators=900, learning_rate=0.024, num_leaves=27,
            min_child_samples=35, colsample_bytree=0.58,
            subsample=0.88, subsample_freq=1, reg_lambda=3.0,
            reg_alpha=0.12, max_bin=255, n_jobs=N_JOBS,
            random_state=seed, verbosity=-1,
        )
    if algorithm == "XGBoost":
        return XGBClassifier(
            n_estimators=850,
            learning_rate=0.035 if objective == "known" else 0.025,
            max_depth=4, min_child_weight=3, subsample=0.88,
            colsample_bytree=0.63, reg_lambda=2.5, reg_alpha=0.2,
            tree_method="hist", max_bin=256, eval_metric="auc",
            n_jobs=N_JOBS, random_state=seed,
        )
    if algorithm == "CatBoost":
        return CatBoostClassifier(
            iterations=1100, learning_rate=0.055, depth=6,
            l2_leaf_reg=5, random_strength=0.8,
            loss_function="Logloss", verbose=False,
            thread_count=N_JOBS, random_seed=seed,
            allow_writing_files=False,
        )
    raise ValueError(algorithm)


def model_predictions(data, cutoff, x_eval):
    out = {}
    for objective in ("known", "background"):
        rows = data["steps"] <= cutoff
        if objective == "known":
            rows &= data["known"]
        for algorithm in ALGORITHMS:
            model = model_for(algorithm, objective)
            model.fit(data["x_train"][rows], data["target"][rows])
            out[(objective, algorithm)] = model.predict_proba(x_eval)[:, 1]
            print(f"Training cutoff {cutoff}: {objective} / {algorithm}", flush=True)
    return out




def model_predictions_saved(model_dir, evaluation_features):
    from lightgbm import Booster
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    result = {}
    for objective in ("known", "background"):
        for algorithm in ALGORITHMS:
            name = f"{objective}_{algorithm.lower()}"
            if algorithm == "LightGBM":
                model = Booster(model_file=str(model_dir / f"{name}.txt"))
                values = model.predict(evaluation_features, num_threads=N_JOBS)
            elif algorithm == "XGBoost":
                model = XGBClassifier()
                model.load_model(str(model_dir / f"{name}.json"))
                values = model.predict_proba(evaluation_features)[:, 1]
            else:
                model = CatBoostClassifier()
                model.load_model(str(model_dir / f"{name}.cbm"))
                values = model.predict_proba(evaluation_features)[:, 1]
            result[(objective, algorithm)] = values
    return result

def safe_logit(values):
    return logit(np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5))


def ensemble_probabilities(predictions):
    probabilities = {}
    for objective in ("known", "background"):
        probabilities[objective] = sum(
            weight * predictions[(objective, algorithm)]
            for algorithm, weight in zip(ALGORITHMS, ALGORITHM_WEIGHTS)
        )

    known_z = safe_logit(probabilities["known"])
    background_z = safe_logit(probabilities["background"])
    fixed = expit(0.20 * known_z + 0.80 * background_z)

    background_weight = np.clip(
        0.80 - 0.08 * np.tanh((known_z - background_z) / 3.0),
        0.60, 0.95,
    )
    adaptive = expit(
        (1.0 - background_weight) * known_z + background_weight * background_z
    )
    dynamic = expit(0.60 * safe_logit(adaptive) + 0.40 * safe_logit(fixed))
    return fixed, dynamic


def validation_for_dynamic(data):
    results = []
    for cutoff, start, end in VALIDATION_WINDOWS:
        mask = data["known"] & (data["steps"] >= start) & (data["steps"] <= end)
        scores = model_predictions(data, cutoff, data["x_train"][mask])
        fixed, dynamic = ensemble_probabilities(scores)
        y = data["target"][mask]
        results.append({
            "train_through": cutoff,
            "validation_steps": f"{start}-{end}",
            "n": int(mask.sum()),
            "illicit_n": int(y.sum()),
            "fixed_auc": roc_auc_score(y, fixed),
            "dynamic_auc": roc_auc_score(y, dynamic),
            "dynamic_average_precision": average_precision_score(y, dynamic),
        })
    return pd.DataFrame(results)


def make_submission(data, probabilities, path):
    result = pd.DataFrame({
        "index": data["test"]["index"].to_numpy(),
        "target": np.asarray(probabilities),
    })
    if not result["index"].equals(data["template"]["index"]):
        raise ValueError("Submission index order differs from the sample")
    if len(result) != len(data["test"]) or not result["target"].between(0, 1).all():
        raise ValueError("Invalid submission predictions")
    if not np.isfinite(result["target"]).all():
        raise ValueError("Submission predictions contain non-finite values")
    result.to_csv(path, index=False)
    print(f"Saved {path} ({len(result):,} rows)")
    return result


def main():
    root = Path(__file__).resolve().parent
    data = load_data(os.environ.get("BITCOIN_DATA_DIR", root / "data"))
    if os.environ.get("SKIP_VALIDATION", "0") != "1":
        validation = validation_for_dynamic(data)
        validation.to_csv(root / "validation_dynamic.csv", index=False)
        print(validation.round(6).to_string(index=False))
    model_dir = root / "models"
    if os.environ.get("RETRAIN_FROM_SCRATCH", "0") == "1":
        predictions = model_predictions(data, int(data["steps"].max()), data["x_test"])
    else:
        if not model_dir.is_dir():
            raise FileNotFoundError("Saved models are missing. Restore models/ or set RETRAIN_FROM_SCRATCH=1 to train again.")
        predictions = model_predictions_saved(model_dir, data["x_test"])
        print("Using the saved fitted models for final inference")
    _, dynamic = ensemble_probabilities(predictions)
    make_submission(data, dynamic, root / "dynamic_adaptive_submission_7.csv")


if __name__ == "__main__":
    main()
