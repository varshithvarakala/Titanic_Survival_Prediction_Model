# Titanic Survival Prediction

A classifier benchmarking project built on the classic Titanic dataset. The pipeline cleans and encodes raw passenger records, then trains and compares ten different machine learning algorithms to predict survival outcome.

## Project Overview

The Titanic dataset is a standard benchmark for binary classification: given a passenger's class, age, family size, sex, and embarkation point, predict whether they survived. This project treats it as a model comparison exercise — rather than committing to a single algorithm, it systematically evaluates a wide range of classifiers under identical preprocessing to identify which family of models (linear, tree-based, distance-based, boosting) performs best on this data.

## Dataset

- **Source:** `Titanic-dataset.csv`
- **Rows / Columns:** 891 passengers, 12 original columns
- **Target:** `Survived` (0 = did not survive, 1 = survived)

## Data Cleaning & Preprocessing

**Missing values** (identified via `df.isnull().sum()`):

| Column | Missing Count |
|---|---|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

**Handling strategy:**
- `Age` — missing values filled with the column median.
- `Cabin` — dropped entirely; missing in ~77% of rows, too sparse to impute reliably.
- `Name`, `Ticket`, `PassengerId` — dropped as identifier/free-text fields with no direct predictive value in raw form.
- `Embarked` — the 2 rows with missing values are dropped rather than imputed, since the loss is negligible relative to the dataset size.

**Encoding:**
- `Sex` and `Embarked` are one-hot encoded via `pd.get_dummies(..., drop_first=True)`, which avoids the dummy variable trap by dropping one category per feature.

## Exploratory Data Analysis

**Missing values by column**

![Missing Values by Column](reports/01_missing_values.png)

**Survival rate by sex**

![Survival Rate by Sex](reports/02_survival_by_sex.png)

**Survival rate by passenger class**

![Survival Rate by Passenger Class](reports/03_survival_by_pclass.png)

## Modeling

Ten classifiers were trained and evaluated on the same train/test split (80/20, `random_state=42`). Distance-based and linear models (SVM, KNN, Logistic Regression) were trained on `StandardScaler`-scaled features, fit on the training set only; tree-based and boosting models were trained on the raw encoded features.

| Model | Accuracy |
|---|---|
| Support Vector Machine (RBF) | 0.8258 |
| Gradient Boosting | 0.8146 |
| LightGBM | 0.8090 |
| AdaBoost | 0.8090 |
| XGBoost | 0.7978 |
| K-Nearest Neighbors | 0.7921 |
| Logistic Regression | 0.7809 |
| Random Forest | 0.7753 |
| Naive Bayes | 0.7753 |
| Decision Tree | 0.7472 |


## Results

**Accuracy comparison across all models**

![Classifier Accuracy Comparison](reports/04_model_accuracy_comparison.png)

**Confusion matrix — best-performing model**

![Confusion Matrix - Best Model](reports/05_confusion_matrix_best_model.png)

## Key Insights

1. **Non-linear models win.** SVM (RBF kernel) and boosting methods (AdaBoost, XGBoost, Gradient Boosting) consistently outperform linear and single-tree baselines, suggesting survival likelihood depends on non-linear interactions between features (e.g., class and sex together) rather than any single additive effect.
2. **Sex and class are dominant signals.** The survival-rate breakdowns by sex and passenger class show sharp splits on their own, well before any model is trained — consistent with the historical "women and children first, first class priority" evacuation pattern.
3. **Scaling methodology matters.** Fitting a scaler on the training set only (not on train+test independently) avoids subtly leaking test-set distribution information into the model — worth double-checking in any pipeline that scales features.

## Repository Structure

```text
├── data/                   # Titanic-dataset.csv
├── notebooks/              # titanic_model_comparison.ipynb
├── reports/                # Saved plots 
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/titanic_model_comparison.ipynb
```

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `matplotlib`

