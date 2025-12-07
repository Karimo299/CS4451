"""
experiments.py

Run cross-validated hyperparameter search and evaluation on Brain_Tumor.csv.

- Loads Brain_Tumor.csv
- Cleans infinities / NaNs
- Splits into train/test
- Uses StratifiedKFold + GridSearchCV to find best hyperparameters
- Reports CV performance and test performance (accuracy, precision, recall, F1, ROC AUC)
- Plots and saves ROC curve (roc_curve.png) for the test set
- Plots and saves Precision–Recall curve (pr_curve.png) for the test set
- Prints feature importance based on logistic regression coefficients
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CSV_PATH = "Brain_Tumor.csv"   # Make sure this file is in the same folder
TARGET_COL = "Class"           # Your label column
RANDOM_STATE = 42
METRIC_DIGITS = 4


# ---------------------------------------------------------------------
# SMALL PRINT HELPERS
# ---------------------------------------------------------------------

def print_section(title: str, char: str = "="):
    line = char * max(len(title), 60)
    print("\n" + line)
    print(title)
    print(line)


def print_subsection(title: str):
    line = "-" * max(len(title), 40)
    print("\n" + title)
    print(line)


# ---------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------

def load_and_clean_data(csv_path: str, target_col: str):
    """Load Brain_Tumor.csv and return clean numeric X, y."""
    df = pd.read_csv(csv_path)
    print_section(f"DATA LOADING: {csv_path}", "=")
    print(f"Raw shape: {df.shape}")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep only numeric columns for features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    print_subsection("Feature Columns (numeric only)")
    print(f"{len(numeric_cols)} features used:")
    print(numeric_cols)

    # Replace inf / -inf with NaN then impute
    X = X.replace([np.inf, -np.inf], np.nan)
    num_inf = X.isna().sum().sum()
    if num_inf > 0:
        print(f"\n[INFO] Replacing {num_inf} inf / NaN values with column medians.")

    X = X.fillna(X.median(numeric_only=True))

    # Drop rows where target is missing
    mask_valid = y.notna()
    X = X.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)

    print_subsection("Cleaned Data Summary")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nClass distribution:")
    print(y.value_counts())

    return X, y


# ---------------------------------------------------------------------
# MODEL TRAINING & HYPERPARAMETER SEARCH
# ---------------------------------------------------------------------

def build_pipeline():
    """Return a base pipeline: StandardScaler + LogisticRegression."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    return pipe


def get_param_grid():
    """Hyperparameter grid for LogisticRegression."""
    param_grid = [
        {
            "clf__penalty": ["l1"],
            "clf__solver": ["liblinear", "saga"],
            "clf__C": [0.01, 0.1, 1, 10, 100],
        },
        {
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "saga"],
            "clf__C": [0.01, 0.1, 1, 10, 100],
        },
    ]
    return param_grid


def run_hyperparameter_search(X_train, y_train):
    """

    """
    print_section("MODEL SELECTION: Logistic Regression")

    pipe = build_pipeline()
    param_grid = get_param_grid()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,   # keep output clean
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_roc_auc = grid.best_score_

    print_subsection("Best Hyperparameters (GridSearchCV, scoring = ROC AUC)")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"\nBest mean CV ROC AUC: {best_roc_auc:.{METRIC_DIGITS}f}")

    # Additional CV with best model for other metrics
    print_subsection("Cross-Validation Performance (Best Model)")
    for metric in ["accuracy", "f1", "roc_auc"]:
        scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=cv,
            scoring=metric,
            n_jobs=-1,
        )
        mean = scores.mean()
        spread = 2 * scores.std()
        print(
            f"{metric.upper():10s}: "
            f"{mean:.{METRIC_DIGITS}f} ± {spread:.{METRIC_DIGITS}f} (mean ± 2*std)"
        )

    return best_model, best_params, best_roc_auc


# ---------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------

def print_feature_importance(model, feature_names):
    """
    Print sorted feature importances based on logistic regression coefficients.
    Positive coef -> increases log-odds of Class=1 (tumor),
    Negative coef -> increases log-odds of Class=0 (no tumor).
    """
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]  # binary classification: one row

    importance = np.abs(coefs)
    sorted_idx = np.argsort(importance)[::-1]

    rows = []
    for idx in sorted_idx:
        fname = feature_names[idx]
        c = coefs[idx]
        effect = "↑ tumor probability" if c > 0 else "↓ tumor probability" if c < 0 else "no effect"
        rows.append(
            {
                "Feature": fname,
                "Coefficient": c,
                "|Coefficient|": abs(c),
                "Effect (sign)": effect,
            }
        )

    df_imp = pd.DataFrame(rows)

    print_section("FEATURE IMPORTANCE (Logistic Regression Coefficients)", "=")
    print(df_imp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ---------------------------------------------------------------------
# FINAL TEST EVALUATION + ROC & PR PLOTS
# ---------------------------------------------------------------------

def evaluate_on_test(model, X_test, y_test):
    """Evaluate the chosen model on the held-out test set and plot ROC + PR curves."""
    print_section("TEST SET EVALUATION", "=")

    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print_subsection("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print_subsection("Classification Report")
    print(classification_report(y_test, y_pred, digits=4))

    print_subsection("Scalar Metrics")
    print(f"Accuracy : {acc:.{METRIC_DIGITS}f}")
    print(f"Precision: {prec:.{METRIC_DIGITS}f}")
    print(f"Recall   : {rec:.{METRIC_DIGITS}f}")
    print(f"F1-score : {f1:.{METRIC_DIGITS}f}")
    if roc is not None:
        print(f"ROC AUC  : {roc:.{METRIC_DIGITS}f}")
    else:
        print("ROC AUC  : (not available, model has no proba/score output)")

    # ---- ROC curve plotting ----
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc:.{METRIC_DIGITS}f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Brain Tumor Classifier")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=300)
        plt.close()
        print("\n[PLOT] Saved ROC curve to roc_curve.png")

        # ---- Precision–Recall curve plotting ----
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall_vals, precision_vals, label="Precision–Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve - Brain Tumor Classifier")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("pr_curve.png", dpi=300)
        plt.close()
        print("[PLOT] Saved Precision–Recall curve to pr_curve.png")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    # 1. Load data
    X, y = load_and_clean_data(CSV_PATH, TARGET_COL)
    feature_names = X.columns.tolist()

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print_section("TRAIN/TEST SPLIT", "=")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    # 3. Hyperparameter search with cross-validation
    best_model, best_params, best_cv_score = run_hyperparameter_search(
        X_train, y_train
    )

    # 4. Final test evaluation + ROC & PR curves
    test_metrics = evaluate_on_test(best_model, X_test, y_test)

    # 5. Feature importance
    print_feature_importance(best_model, feature_names)

    # 6. Final summary
    print_section("FINAL SUMMARY", "=")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV ROC AUC : {best_cv_score:.{METRIC_DIGITS}f}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        if v is None:
            print(f"  {k:10s}: (n/a)")
        else:
            print(f"  {k:10s}: {v:.{METRIC_DIGITS}f}")
    print()


if __name__ == "__main__":
    main()
