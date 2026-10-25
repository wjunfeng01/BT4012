# BT4012 – Bitcoin Transaction Classification

This project uses pretrained machine learning models to classify Bitcoin transactions and generate two prediction files for the Kaggle competition.

## 1. Requirements

- Python 3.13
- Required Python packages listed in `requirements.txt`
- Competition dataset

## 2. Installation

Open a terminal in the project folder and install the required packages:

```bash
py -3.13 -m pip install -r requirements.txt
```

## 3. Dataset Setup

Place the competition dataset ZIP file or extracted CSV files inside the `data/` folder.

The required files are:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The `models/` folder must also be retained, as it contains the pretrained models used to generate predictions.

## 4. Running the Code

### Option A: Jupyter Notebook

1. Open `e1122936.ipynb` in VS Code or Jupyter Notebook.
2. Select the Python 3.13 kernel.
3. Run all cells in order.

### Option B: Python Script

Run the following command from the project directory:

```bash
py -3.13 dynamic_plus_guarded.py
```

This generates both prediction files using the pretrained models.

## 5. Output Files

The code generates two CSV files in the project directory:

- `dynamic_adaptive_submission_7.csv`
- `dynamic_plus_guarded_8.csv`

Each file contains the transaction index and its predicted illicit transaction probability.

## 6. Project Files

| File / Folder | Description |
|---|---|
| `e1122936.ipynb` | Jupyter notebook for running the project |
| `dynamic_adaptive.py` | Data processing and adaptive ensemble implementation |
| `dynamic_plus_guarded.py` | Guarded ensemble and prediction generation |
| `models/` | Saved trained models |
| `data/` | Competition dataset |
| `requirements.txt` | Python dependencies |

**Note:** The dataset is not included in the repository and must be added to the `data/` folder before running the code.