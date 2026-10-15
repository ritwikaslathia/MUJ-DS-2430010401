# MUJ-DS-2430010401

| Field | Details |
|---|---|
| **Name** | Ritwika Slathia |
| **Registration Number** | 2430010401 |
| **Branch** | Computer Science (Data Science) |
| **Batch** | E |
| **Project Title** | Diabetes Prediction using Deep Learning: Neural Networks vs Transformer (PIMA Indians Dataset) |
| **GitHub Username** | ritwikaslathia |
| **Training Program** | B.Tech in Computer Science (Data Science), Manipal University Jaipur |
| **Instructor** | Sandeep (added as collaborator) |

---

## Project Overview

This project predicts whether a patient has diabetes from 8 medical measurements (glucose, BMI, age, etc.). It compares a baseline neural network, a regularized neural network, and a small Transformer, and includes hyperparameter comparison, an ablation study (L2 and Dropout), and classification-threshold tuning.

## Dataset

- **Name:** PIMA Indians Diabetes Dataset
- **Samples:** 767 loaded in the notebook
- **Features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** `Outcome` (0 = No diabetes, 1 = Diabetes)
- **Class balance:** 500 No diabetes vs 267 Diabetes

## Methodology

| Step | What was done |
|---|---|
| 1. Split | 70% train (536) / 15% validation (115) / 15% test (116), stratified, `random_state=42` |
| 2. Preprocessing | `StandardScaler` fitted on training data only (no data leakage) |
| 3. Baseline NN | Dense(16, ReLU) → Dense(1, Sigmoid), 161 parameters |
| 4. Proposed NN | Dense(32) → Dropout(0.2) → Dense(16) → Dropout(0.2) → Dense(1), L2(0.001), 833 parameters |
| 5. Training | Adam, binary cross-entropy, batch size 16, EarlyStopping on `val_loss` |
| 6. Hyperparameter comparison | 5 configurations (hidden units 16/32/64, dropout 0.1/0.2/0.3) |
| 7. Ablation study | Basic vs L2 only vs Dropout only vs L2 + Dropout |
| 8. Threshold tuning | Threshold searched on validation set (best = 0.28) |
| 9. Final NN | Dense(64) + BatchNorm + Dropout → Dense(32) + BatchNorm + Dropout → Dense(16) → Dense(1) |
| 10. Transformer | Each feature treated as a token → Multi-Head Attention (2 heads) → feed-forward → pooling → Dense, 2,545 parameters |

## Results (Test Set, 116 samples)

| Model | Accuracy | Precision | Recall | F1 | Specificity | AUC |
|---|---|---|---|---|---|---|
| Baseline NN (161 params) | 73.28% | 0.6800 | 0.425 | 0.5231 | 0.8947 | 0.8033 |
| Proposed NN (833 params) | 72.41% | 0.6176 | 0.525 | 0.5676 | 0.8289 | 0.8230 |
| Proposed NN, threshold 0.28 | 73.28% | 0.5818 | 0.800 | 0.6737 | 0.6974 | 0.8230 |
| Final NN (BatchNorm) | 72.41% | 0.6429 | 0.450 | 0.5294 | 0.8684 | 0.8181 |
| Transformer (2,545 params) | 69.83% | 0.5862 | 0.425 | 0.4928 | – | 0.7405 |

### Hyperparameter Comparison

| Hidden Units | Dropout | Test Accuracy | AUC |
|---|---|---|---|
| 16 | 0.2 | 74.14% | 0.8243 |
| 32 | 0.2 | 69.83% | 0.8227 |
| 64 | 0.2 | 71.55% | 0.8224 |
| **32** | **0.1** | **75.00%** | **0.8227** |
| 32 | 0.3 | 71.55% | 0.7970 |

### Ablation Study

| Model | Accuracy | AUC |
|---|---|---|
| Basic | 73.28% | 0.8273 |
| L2 Only | 77.59% | 0.8447 |
| Dropout Only | 71.55% | 0.8089 |
| L2 + Dropout | 72.41% | 0.8043 |

### Key Findings

- The Proposed NN has the best AUC (0.823) of the basic models, so it ranks patients better even though its accuracy at threshold 0.5 is similar to the baseline.
- **Lowering the threshold to 0.28 raised recall from 52.5% to 80.0%** while accuracy rose to 73.28%. This matters in medical screening, where missing a diabetic patient is costlier than a false alarm. The trade-off is lower specificity (69.7%).
- The Transformer scored lowest (69.83% accuracy, AUC 0.7405). With only 8 features and 536 training rows, attention gives no advantage over a small feed-forward network.
- In the ablation study, L2 only scored best (77.59%), while Dropout only was lowest.
- With 116 test samples, one patient is about 0.9%, so differences of 2-3% are within noise. Results also shifted between separate runs of the notebook (for example, L2 only scored 68.10% in an earlier run), so conclusions should be treated as indicative, not definitive.

### Limitations

- The CSV has no header row, so the first record was read as the header (767 of 768 samples used).
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically impossible and were not treated as missing values.
- Small dataset and a single train/validation/test split.

## Repository Structure

```
MUJ-DS-2430010401/
├── README.md
├── requirements.txt
├── .gitignore
├── assignments/       # Course assignments
├── notebooks/         # dlproject.ipynb (main project notebook)
├── code/              # Python scripts
├── resources/         # Dataset (pima-indians-diabetes.data.csv), references
├── results/           # Saved figures, metric CSVs, trained models
├── presentations/     # Project presentation
└── capstone/          # Capstone project work
```

## Installation & How to Run

### Option 1: Google Colab (easiest)
1. Open `notebooks/dlproject.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells. When asked, upload `pima-indians-diabetes.data.csv` (kept in `resources/`).

### Option 2: Run locally
```bash
git clone https://github.com/ritwikaslathia/MUJ-DS-2430010401.git
cd MUJ-DS-2430010401
pip install -r requirements.txt
jupyter notebook notebooks/dlproject.ipynb
```
> The notebook uses `google.colab.files.upload()` to load data. Locally, replace it with `df = pd.read_csv("resources/pima-indians-diabetes.data.csv", header=None)`.

## Output Files (saved to `results/`)

Training and validation accuracy/loss plots, confusion matrix, ROC curve, `performance_metrics.csv`, `hyperparameter_comparison.csv`, `ablation_study.csv`, `final_results.csv`, `computational_cost.csv`, `test_predictions.csv`, `transformer_model_comparison.csv`, and the saved `.keras` models.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Weekly Progress Log

| Week | Work Done |
|---|---|
| 1 | Repository setup, dataset exploration |
| 2 | Preprocessing, baseline model |
| 3 | Improved model, hyperparameter comparison, ablation study |
| 4 | Threshold tuning, Transformer model, final evaluation, documentation |


## Links

- **Presentation:** `presentations/`

