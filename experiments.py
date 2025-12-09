"""
experiments.py

Goal: For the Kaggle Brain Tumor dataset, build the 7 feature subsets (D1–D7)
and train/tune *logistic regression* on each, then compare performance.

Datasets (all aligned over the same N = 3762 samples):

D1: First-order only (tabular)
    Features: Mean, Variance, Standard Deviation, Skewness, Kurtosis

D2: Second-order only (tabular)
    Features: Contrast, Energy, ASM, Entropy, Homogeneity,
              Dissimilarity, Correlation, Coarseness

D3: All tabular features
    First-order + second-order (13 total)

D4: Image only
    64x64 grayscale MRI → flattened vector of length 4096

D5: Image + first-order
    Concatenate D4 pixels with D1 tabular features

D6: Image + second-order
    Concatenate D4 pixels with D2 tabular features

D7: Image + all tabular
    Concatenate D4 pixels with D3 tabular features

Model: Logistic Regression (with StandardScaler in a Pipeline)
- Hyperparameters tuned via GridSearchCV on the training set
- Scoring = accuracy (matches methodology text)
- Results summarized via test accuracy and macro-F1.

At the end we also generate a bar chart that compares
Test Accuracy and Test Macro F1 across D1–D7.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CSV_PATH = "data/Brain Tumor.csv"
IMAGES_DIR = "data/images"
TARGET_COL = "Class"
IMAGE_COL = "Image"

RANDOM_STATE = 42
METRIC_DIGITS = 4
IMAGE_SIZE = (64, 64)  # resized MRI slices
IMAGE_FEATURES_PATH = f"cache_image_features_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.npy"

FIRST_ORDER_COLS = [
    "Mean",
    "Variance",
    "Standard Deviation",
    "Skewness",
    "Kurtosis",
]

SECOND_ORDER_COLS = [
    "Contrast",
    "Energy",
    "ASM",
    "Entropy",
    "Homogeneity",
    "Dissimilarity",
    "Correlation",
    "Coarseness",
]


# ---------------------------------------------------------------------
# PRINT HELPERS
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
# DATA LOADING & TABULAR PREPROCESSING
# ---------------------------------------------------------------------

def load_base_dataframe() -> pd.DataFrame:
    """Load Brain_Tumor.csv and sanity check required columns."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV '{CSV_PATH}' not found. Put Brain_Tumor.csv in the same folder "
            f"as experiments.py (currently: {os.getcwd()})."
        )
    df = pd.read_csv(CSV_PATH)
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    if IMAGE_COL not in df.columns:
        raise KeyError(
            f"Image column '{IMAGE_COL}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def preprocess_tabular(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Clean and extract tabular features.

    - Drop target and image columns from X.
    - Keep only numeric feature columns.
    - Replace +/-inf with NaN and median-impute per numeric column.
    - Drop rows where the target is NaN.

    Returns:
      X_tab       : cleaned numeric features (DataFrame)
      y           : labels (Series)
      feat_names  : list of feature names
      df_clean    : df filtered to rows where y is not NaN,
                    indices aligned with X_tab and y
    """
    df_clean = df.copy()

    y = df_clean[TARGET_COL]
    X = df_clean.drop(columns=[TARGET_COL, IMAGE_COL])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # Replace inf with NaN, then fill with column medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    df_clean = df_clean.loc[mask].reset_index(drop=True)

    return X, y, numeric_cols, df_clean


def summarize_tabular(X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
    """Print a human-readable summary of the tabular dataset."""
    print_section("TABULAR-ONLY DATASET (from Brain_Tumor.csv)", "=")

    print_subsection("Numeric feature columns")
    print(f"{len(feature_names)} features used:")
    print(feature_names)

    print_subsection("Cleaned tabular data summary")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nClass distribution:")
    print(y.value_counts())


# ---------------------------------------------------------------------
# IMAGE LOADING & FEATURE MATRIX
# ---------------------------------------------------------------------

def load_and_preprocess_image(path: str, size=(64, 64)) -> np.ndarray:
    """Load an image, convert to grayscale, resize, flatten to 1D float32 array."""
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize 0–1
    return arr.flatten()


def resolve_image_path(raw_name: str) -> str | None:
    """
    Given something like 'Image1' from the CSV, try to find the actual file
    in IMAGES_DIR. We try common extensions (.jpg, .jpeg, .png).
    """
    base = os.path.basename(str(raw_name)).strip()
    root, ext = os.path.splitext(base)
    candidates = []
    if ext:
        candidates.append(base)
    else:
        for e in [".jpg", ".jpeg", ".png"]:
            candidates.append(root + e)

    for filename in candidates:
        full = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(full):
            return full
    return None


def build_image_features_matrix(df_clean: pd.DataFrame) -> np.ndarray:
    """
    Build an (N, D_img) matrix of flattened image pixels corresponding
    row-by-row to df_clean.

    Uses a simple .npy cache so we don't recompute features every run.
    """
    expected_n = len(df_clean)
    expected_d = IMAGE_SIZE[0] * IMAGE_SIZE[1]

    if os.path.exists(IMAGE_FEATURES_PATH):
        print(f"[INFO] Loading cached image features from {IMAGE_FEATURES_PATH}")
        X_img = np.load(IMAGE_FEATURES_PATH)
        if X_img.shape == (expected_n, expected_d):
            return X_img
        print(
            f"[WARN] Cached features shape {X_img.shape} does not match "
            f"expected {(expected_n, expected_d)}. Recomputing..."
        )

    print("[INFO] Computing image features (once) ...")
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Images folder '{IMAGES_DIR}' not found. "
            "Put the 'images' folder next to experiments.py."
        )

    image_names = df_clean[IMAGE_COL].astype(str).reset_index(drop=True)
    feats: List[np.ndarray] = []

    for raw_name in image_names:
        img_path = resolve_image_path(raw_name)
        if img_path is None:
            raise RuntimeError(
                f"Could not find a file for image ID '{raw_name}' in '{IMAGES_DIR}'. "
                "Make sure files are named like 'Image1.jpg', 'Image2.jpg', ..."
            )
        feats.append(load_and_preprocess_image(img_path, size=IMAGE_SIZE))

    X_img = np.vstack(feats).astype(np.float32)
    if X_img.shape != (expected_n, expected_d):
        raise RuntimeError(
            f"Built image feature matrix with shape {X_img.shape}, "
            f"expected {(expected_n, expected_d)}."
        )

    np.save(IMAGE_FEATURES_PATH, X_img)
    print(f"[INFO] Saved image features to {IMAGE_FEATURES_PATH}")
    return X_img


# ---------------------------------------------------------------------
# BUILD ALL 7 DATASETS (D1–D7)
# ---------------------------------------------------------------------

def build_all_datasets(
    X_tab_np: np.ndarray,
    feat_names: List[str],
    X_img: np.ndarray,
    y_np: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    """
    Construct the 7 datasets D1–D7 as defined in the methodology.

    Returns a dict mapping dataset ID -> dict with:
      - 'X': feature matrix
      - 'y': labels
      - 'pretty': human-readable description
    """
    name_to_idx = {name: i for i, name in enumerate(feat_names)}

    try:
        idx_first = [name_to_idx[c] for c in FIRST_ORDER_COLS]
        idx_second = [name_to_idx[c] for c in SECOND_ORDER_COLS]
    except KeyError as e:
        raise KeyError(
            f"Required feature column {str(e)} not found in feat_names. "
            f"Got feat_names={feat_names}"
        )

    X_first = X_tab_np[:, idx_first]
    X_second = X_tab_np[:, idx_second]
    X_all_tab = X_tab_np  # 13 features

    datasets = {
        "D1_first_order": {
            "X": X_first,
            "y": y_np,
            "pretty": "D1 – Tabular: First-order only",
        },
        "D2_second_order": {
            "X": X_second,
            "y": y_np,
            "pretty": "D2 – Tabular: Second-order only",
        },
        "D3_all_tabular": {
            "X": X_all_tab,
            "y": y_np,
            "pretty": "D3 – Tabular: All 13 features",
        },
        "D4_image_only": {
            "X": X_img,
            "y": y_np,
            "pretty": "D4 – Image only (64x64 grayscale)",
        },
        "D5_image_plus_first": {
            "X": np.concatenate([X_img, X_first], axis=1),
            "y": y_np,
            "pretty": "D5 – Image + first-order features",
        },
        "D6_image_plus_second": {
            "X": np.concatenate([X_img, X_second], axis=1),
            "y": y_np,
            "pretty": "D6 – Image + second-order features",
        },
        "D7_image_plus_all": {
            "X": np.concatenate([X_img, X_all_tab], axis=1),
            "y": y_np,
            "pretty": "D7 – Image + all tabular features",
        },
    }

    print_subsection("Dataset shapes (D1–D7)")
    for key, info in datasets.items():
        Xk: np.ndarray = info["X"]  # type: ignore
        print(f"{key}: X shape = {Xk.shape}, y shape = {info['y'].shape}")

    return datasets


# ---------------------------------------------------------------------
# MODEL & GRID SEARCH (LOGISTIC REGRESSION)
# ---------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Logistic regression pipeline with standardization."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def get_param_grid(full: bool) -> List[Dict[str, object]]:
    """
    Return the hyperparameter grid.

    full = True  -> larger grid (L1/L2, multiple solvers), used for D1–D3.
    full = False -> lighter grid (L2 + lbfgs only), used for image-based D4–D7.
    """
    Cs = [0.01, 0.1, 1, 10]

    if full:
        # Explore penalties and solvers more thoroughly on lower-dimensional tabular subsets.
        return [
            {"clf__penalty": ["l2"], "clf__solver": ["lbfgs"],     "clf__C": Cs},
            {"clf__penalty": ["l2"], "clf__solver": ["liblinear"], "clf__C": Cs},
            {"clf__penalty": ["l1"], "clf__solver": ["liblinear"], "clf__C": Cs},
            {"clf__penalty": ["l1"], "clf__solver": ["saga"],      "clf__C": Cs},
        ]
    else:
        # High-dimensional image-based subsets: keep it computationally manageable.
        return [
            {"clf__penalty": ["l2"], "clf__solver": ["lbfgs"], "clf__C": Cs},
        ]


def run_hyperparameter_search(ds_key: str, X_train, y_train, label: str):
    print_section(f"MODEL SELECTION – {label}", "=")

    pipe = build_pipeline()

    # Use full grid for tabular-only; lighter grid for image-based datasets
    tabular_only_keys = {"D1_first_order", "D2_second_order", "D3_all_tabular"}
    use_full_grid = ds_key in tabular_only_keys

    param_grid = get_param_grid(full=use_full_grid)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",        # matches methodology
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    best_params = gs.best_params_
    best_acc = gs.best_score_

    print_subsection("Best hyperparameters")
    print(best_params)
    print(f"Best CV Accuracy: {best_acc:.{METRIC_DIGITS}f}")

    return best_model, best_params, best_acc


# ---------------------------------------------------------------------
# EVALUATION (NO ROC / NO CONFUSION MATRIX PLOTS)
# ---------------------------------------------------------------------

def evaluate_on_test(model, X_test, y_test, file_label: str, pretty_label: str):
    print_section(f"TEST SET EVALUATION – {pretty_label}", "=")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_bin = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print_subsection("Scalar metrics")
    print(f"Accuracy   : {acc:.{METRIC_DIGITS}f}")
    print(f"Precision  : {prec:.{METRIC_DIGITS}f}")
    print(f"Recall     : {rec:.{METRIC_DIGITS}f}")
    print(f"F1 (binary): {f1_bin:.{METRIC_DIGITS}f}")
    print(f"F1 (macro) : {f1_macro:.{METRIC_DIGITS}f}")

    # Just for textual insight, we can still print the confusion matrix and classification report,
    # but we DO NOT plot anything or save ROC/PR/confusion matrix images.
    cm = confusion_matrix(y_test, y_pred)
    print_subsection("Confusion matrix")
    print(cm)

    print_subsection("Classification report")
    print(classification_report(y_test, y_pred, digits=METRIC_DIGITS))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_binary": f1_bin,
        "f1_macro": f1_macro,
    }


# ---------------------------------------------------------------------
# SUMMARY BAR CHART (BEST VISUAL FOR THE PAPER)
# ---------------------------------------------------------------------
def plot_summary_bar_chart(df_comp: pd.DataFrame, out_dir: Path):
    """
    Bar chart comparing Test Accuracy and Test Macro F1 across D1–D7.

    X-axis: D1..D7 only (short, non-overlapping).
    Full text descriptions stay in the summary table / paper caption.
    """
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / "logreg_D1_D7_bar_comparison.png"

    # Short labels (D1..D7)
    labels = df_comp["Dataset"].tolist()          # e.g. "D7_image_plus_all"
    labels_short = [lbl.split("_")[0] for lbl in labels]  # "D7", "D3", ...

    x = np.arange(len(labels_short))
    width = 0.35

    acc_values = df_comp["Test Accuracy"].values
    f1_values = df_comp["Test Macro F1"].values

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_acc = ax.bar(x - width / 2, acc_values, width, label="Accuracy")
    bars_f1 = ax.bar(x + width / 2, f1_values, width, label="Macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Logistic Regression Performance Across D1–D7")
    ax.legend(loc="lower right")

    # Light horizontal grid
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels on top of each bar
    ax.bar_label(bars_acc, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_f1, fmt="%.3f", padding=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print_subsection("Saved comparison bar chart")
    print(f"Bar chart (Accuracy & Macro F1 across D1–D7): {fig_path}")



# ---------------------------------------------------------------------
# MAIN EXPERIMENT LOGIC (LOGISTIC REGRESSION ON D1–D7)
# ---------------------------------------------------------------------

def main():
    # Hard determinism for numpy & Python RNG
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df_raw = load_base_dataframe()

    # 1) Preprocess tabular once and summarize
    X_tab_df, y, feat_names, df_clean = preprocess_tabular(df_raw)
    summarize_tabular(X_tab_df, y, feat_names)

    # Convert to NumPy
    X_tab_np = X_tab_df.to_numpy(dtype=np.float32)
    y_np = y.to_numpy()

    # 2) Build image feature matrix (cached)
    X_img = build_image_features_matrix(df_clean)
    print_subsection("Image feature matrix summary")
    print(f"X_img shape: {X_img.shape}")

    # 3) Construct all 7 datasets
    datasets = build_all_datasets(X_tab_np, feat_names, X_img, y_np)

    # 4) Shared train/test split on indices (same for all D1–D7)
    n = len(y_np)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_np,
        random_state=RANDOM_STATE,
    )

    print_section("TRAIN/TEST SPLIT – Shared indices for all D1–D7", "=")
    print(f"Total samples: {n}")
    print(f"Train size   : {len(train_idx)}")
    print(f"Test size    : {len(test_idx)}")

    # 5) Run logistic regression on each dataset
    all_results = []

    for ds_key, info in datasets.items():
        X_full: np.ndarray = info["X"]  # type: ignore
        y_full: np.ndarray = info["y"]  # type: ignore
        pretty = info["pretty"]         # type: ignore

        print_section(f"==== DATASET {ds_key} / {pretty} ====", "=")
        X_train = X_full[train_idx]
        X_test = X_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]

        model, params, best_acc = run_hyperparameter_search(ds_key, X_train, y_train, pretty)
        metrics = evaluate_on_test(
            model,
            X_test,
            y_test,
            file_label=ds_key,
            pretty_label=pretty,
        )

        row = {
            "Dataset": ds_key,
            "Description": pretty,
            "Test Accuracy": metrics["accuracy"],
            "Test Macro F1": metrics["f1_macro"],
        }
        all_results.append(row)

    # 6) Final summary table over all D1–D7 for logistic regression
    print_section("FINAL COMPARISON – Logistic Regression over D1–D7", "=")
    df_comp = pd.DataFrame(all_results)
    df_comp = df_comp.sort_values(by="Test Accuracy", ascending=False)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "logreg_D1_D7_summary.csv"
    df_comp.to_csv(summary_path, index=False)

    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df_comp.to_string(index=False))

    # Highlight the best model (by accuracy; you can switch to Macro F1 if you want)
    best_row = df_comp.iloc[0]
    print_subsection("Best logistic regression model (by Test Accuracy)")
    print(best_row.to_string())
    print(f"\nSummary saved to: {summary_path}")

    # 7) Plot the comparison bar chart (Accuracy & Macro F1)
    plot_summary_bar_chart(df_comp, out_dir)


if __name__ == "__main__":
    main()
