import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

DATA_CSV_PATH = "dataset/BrainTumorDataset/Brain Tumor.csv"
IMAGE_DIR = "dataset/Brain Tumor/Brain Tumor"


"""
Load the brain tumor dataset from CSV and image files and construct seven
aligned datasets.

The CSV is expected to contain one row per image with:
  - an image filename in the 'Image' column (e.g., 'Image1' or 'Image1.jpg'),
  - a binary label in the 'Class' column (1 = tumor, 0 = non-tumor),
  - five first-order features: Mean, Variance, Standard Deviation, Skewness, Kurtosis,
  - eight second-order (texture) features: Contrast, Energy, ASM, Entropy,
    Homogeneity, Dissimilarity, Correlation, Coarseness.

This function:
  1. Reads the CSV with pandas.
  2. Checks that the expected columns exist.
  3. Builds three tabular feature matrices:
       - full tabular: all feature columns except Image and Class,
       - first-order only: the 5 first-order features,
       - second-order only: the 8 second-order features.
  4. Encodes the Class column as integer labels y (0/1) and stores the original
     label names.
  5. Loads each image file from disk, converts it to grayscale, resizes it to
     `image_size`, and normalizes pixel intensities to [0, 1].
  6. Combines these into seven dataset dictionaries that all share the same
     sample order.

Returns a 7-tuple:
  - dataset_image_plus_tabular: dict with keys
        'images'       -> np.ndarray of shape (N, H, W), float32 in [0, 1]
        'tabular'      -> np.ndarray of shape (N, D_full)
        'y'            -> np.ndarray of shape (N,), int64 labels
        'label_names'  -> array of original label names
        'feature_cols' -> list of all tabular feature column names
  - dataset_tabular_only: dict with keys
        'X'            -> np.ndarray of shape (N, D_full)
        'y'            -> np.ndarray of shape (N,)
        'label_names'  -> array of original label names
        'feature_cols' -> list of all tabular feature column names
  - dataset_first_order_only: dict with keys
        'X'            -> np.ndarray of shape (N, 5)
        'y'            -> np.ndarray of shape (N,)
        'label_names'  -> array of original label names
        'feature_cols' -> list of first-order feature names
  - dataset_second_order_only: dict with keys
        'X'            -> np.ndarray of shape (N, 8)
        'y'            -> np.ndarray of shape (N,)
        'label_names'  -> array of original label names
        'feature_cols' -> list of second-order feature names
  - dataset_image_plus_first_order: dict with keys
        'images'       -> np.ndarray of shape (N, H, W)
        'tabular'      -> np.ndarray of shape (N, 5)
        'y'            -> np.ndarray of shape (N,)
        'label_names'  -> array of original label names
        'feature_cols' -> list of first-order feature names
  - dataset_image_plus_second_order: dict with keys
        'images'       -> np.ndarray of shape (N, H, W)
        'tabular'      -> np.ndarray of shape (N, 8)
        'y'            -> np.ndarray of shape (N,)
        'label_names'  -> array of original label names
        'feature_cols' -> list of second-order feature names
  - dataset_image_only: dict with keys
        'images'       -> np.ndarray of shape (N, H, W)
        'y'            -> np.ndarray of shape (N,)
        'label_names'  -> array of original label names

All seven datasets use the same labels y and the same row ordering.
"""
def load_brain_tumor_datasets(
    csv_path: str = DATA_CSV_PATH,
    image_dir: str = IMAGE_DIR,
    image_size: tuple[int, int] = (64, 64),
):
    df = pd.read_csv(csv_path)

    image_col = "Image"
    label_col = "Class"

    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in {csv_path}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {csv_path}")

    # All tabular feature columns (everything except image name + label)
    feature_cols = [c for c in df.columns if c not in (image_col, label_col)]
    if len(feature_cols) == 0:
        raise ValueError("No tabular feature columns found (only Image and Class present).")

    # Define first-order and second-order feature groups
    first_order_cols = [
        "Mean",
        "Variance",
        "Standard Deviation",
        "Skewness",
        "Kurtosis",
    ]

    second_order_cols = [
        "Contrast",
        "Energy",
        "ASM",
        "Entropy",
        "Homogeneity",
        "Dissimilarity",
        "Correlation",
        "Coarseness",
    ]

    # Sanity: ensure all requested columns exist
    for col in first_order_cols + second_order_cols:
        if col not in df.columns:
            raise ValueError(f"Expected feature column '{col}' not found in {csv_path}")

    # Full tabular, first-order-only, second-order-only
    X_tabular    = df[feature_cols].to_numpy(dtype=np.float32)
    X_first_ord  = df[first_order_cols].to_numpy(dtype=np.float32)
    X_second_ord = df[second_order_cols].to_numpy(dtype=np.float32)

    # Labels
    labels_cat  = df[label_col].astype("category")
    y           = labels_cat.cat.codes.to_numpy(dtype=np.int64)
    label_names = labels_cat.cat.categories.to_numpy()

    # Images
    images = []
    for raw_name in df[image_col].astype(str):
        root, ext = os.path.splitext(raw_name)
        if ext == "":
            fname = raw_name + ".jpg"
        else:
            fname = raw_name

        img_path = os.path.join(image_dir, fname)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("L")
        img = img.resize(image_size)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(img_arr)

    X_images = np.stack(images, axis=0)

    # 1) images + all tabular
    dataset_image_plus_tabular = {
        "images": X_images,
        "tabular": X_tabular,
        "y": y,
        "label_names": label_names,
        "feature_cols": feature_cols,
    }

    # 2) tabular only (all features)
    dataset_tabular_only = {
        "X": X_tabular,
        "y": y,
        "label_names": label_names,
        "feature_cols": feature_cols,
    }

    # 3) first-order only
    dataset_first_order_only = {
        "X": X_first_ord,
        "y": y,
        "label_names": label_names,
        "feature_cols": first_order_cols,
    }

    # 4) second-order only
    dataset_second_order_only = {
        "X": X_second_ord,
        "y": y,
        "label_names": label_names,
        "feature_cols": second_order_cols,
    }

    # 5) images + first-order
    dataset_image_plus_first_order = {
        "images": X_images,
        "tabular": X_first_ord,
        "y": y,
        "label_names": label_names,
        "feature_cols": first_order_cols,
    }

    # 6) images + second-order
    dataset_image_plus_second_order = {
        "images": X_images,
        "tabular": X_second_ord,
        "y": y,
        "label_names": label_names,
        "feature_cols": second_order_cols,
    }

    # 7) images only
    dataset_image_only = {
        "images": X_images,
        "y": y,
        "label_names": label_names,
    }

    return (
        dataset_image_plus_tabular,      # 1
        dataset_tabular_only,            # 2
        dataset_first_order_only,        # 3
        dataset_second_order_only,       # 4
        dataset_image_plus_first_order,  # 5
        dataset_image_plus_second_order, # 6
        dataset_image_only,              # 7
    )

"""
    Create an SVM classifier wrapped in a scikit-learn Pipeline.

    The pipeline has two steps:
      1. StandardScaler: standardizes each feature to zero mean and unit variance.
      2. SVC: RBF-kernel support vector machine with C=1.0 and gamma='scale'.
"""
def make_svm_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
        # add class_weight="balanced" if your classes are imbalanced
    ])

"""
    Train an SVM model on a train/test split and print basic evaluation metrics.

    Steps:
      - Split the data into training and test sets (80/20 split), keeping the
        class proportions the same with stratify=y.
      - Build an SVM pipeline (scaling + RBF SVM) using make_svm_pipeline().
      - Fit the model on the training data.
      - Predict labels for the test data.
      - Compute and print:
          * Accuracy
          * Macro-averaged F1-score
          * Full classification_report with precision/recall/F1 per class.
      - Return the trained Pipeline object so it can be reused later
        (for example, for further analysis or saving to disk).
"""
def train_and_eval_svm(X, y, name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_svm_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1: {f1:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return model

"""
    Run a random-label sanity check to detect possible label leakage or bugs.

    Steps:
      - Shuffle the labels y to break any real relationship between X and y.
      - Split (X, y_shuffled) into train and test sets (80/20) with stratification.
      - Train the usual SVM pipeline on the shuffled labels.
      - Evaluate accuracy on the shuffled test labels and print the result.

    If the pipeline is correct and there is no leakage, the accuracy here should
    be close to the majority-class baseline (around 0.5–0.6 for this dataset),
    not close to the very high accuracy obtained with the true labels. A high
    accuracy on shuffled labels would be a strong warning sign.
"""
def random_label_sanity(X, y, name=""):
    y_shuffled = np.random.permutation(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_shuffled, test_size=0.2, stratify=y_shuffled, random_state=42
    )
    model = make_svm_pipeline()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"[RANDOM LABEL STRICT] {name} accuracy: {acc:.3f}")


'''
    Apply cross-validation

    Used stratified k fold with 3 folds.
    Also added grid search to see which parameters are the best from the grid for each dataset.
    Prints out the best parameters and the best score (the score achieved with those parameters).

    Steps:
    - Build svm pipeline, that is the model
    - Split the data into training and test sets (80/20 split)
    - Now use stratified k fold with 3 folds
    - Make a parameter grid that will be used for grid search, where in it C and gamma take on different potential values, but the kernel will only be rbf
    - Evaluate grid search using the model and skfold
    - Fit the model on the training data only
    - Get the best parameters and score from those parameters to print them out
    - Calculate the accuracy of those parameters on the test set from the original train test split
'''
def cross_validation_svm(X, y, name: str):
    model = make_svm_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    skfold = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    params_grid = {
        'svm__C': [0.01, 1.0, 10.0, 100.0],
        'svm__kernel': ['rbf'],
        'svm__gamma': [0.01, 1.0, 10.0, 100.0],
    }
    grid = GridSearchCV(estimator=model, param_grid=params_grid, cv=skfold)
    grid.fit(X_train, y_train)

    best_score = grid.best_score_
    best_params = grid.best_params_
    print("FOR ", name)
    print("best parameters found: ", best_params)
    print("best score: ", best_score)

    final_score = grid.score(X_test, y_test)
    print("final grid accuracy on test data: ", final_score)


if __name__ == "__main__":
    (
        ds_img_tab,        # images + all tabular
        ds_tab,            # tabular only (all features)
        ds_first,          # first-order only
        ds_second,         # second-order only
        ds_img_first,      # images + first-order
        ds_img_second,     # images + second-order
        ds_img_only,       # images only
    ) = load_brain_tumor_datasets()

    # ------------------------
    # 1) SVM on tabular-only (all 13 features)
    # ------------------------
    X_tab = ds_tab["X"]          # (N, 13)
    y_tab = ds_tab["y"]          # (N,)
    svm_tabular = train_and_eval_svm(
        X_tab,
        y_tab,
        name="SVM (tabular-only)",
    )
    cross_validation_svm(X_tab, y_tab, "Cross-validation (tabular-only)")

    # ------------------------
    # 2) SVM on first-order features only
    #    (Mean, Variance, Standard Deviation, Skewness, Kurtosis)
    # ------------------------
    X_first = ds_first["X"]      # (N, 5)
    y_first = ds_first["y"]
    svm_first = train_and_eval_svm(
        X_first,
        y_first,
        name="SVM (first-order only)",
    )
    cross_validation_svm(X_first, y_first, "SVM (first-order only)")

    # ------------------------
    # 3) SVM on second-order (texture) features only
    #    (Contrast, Energy, ASM, Entropy, Homogeneity, Dissimilarity,
    #     Correlation, Coarseness)
    # ------------------------
    X_second = ds_second["X"]    # (N, 8)
    y_second = ds_second["y"]
    svm_second = train_and_eval_svm(
        X_second,
        y_second,
        name="SVM (second-order only)",
    )
    cross_validation_svm(X_second, y_second, "SVM (second-order only)")

    # ------------------------
    # 4) SVM on images + all tabular features
    # ------------------------
    X_images_all = ds_img_tab["images"]   # (N, H, W)
    X_tab_all    = ds_img_tab["tabular"]  # (N, 13)
    y_img_all    = ds_img_tab["y"]

    N, H, W = X_images_all.shape
    X_img_flat_all = X_images_all.reshape(N, H * W)             # (N, H*W)
    X_img_plus_tab = np.concatenate([X_img_flat_all, X_tab_all], axis=1)

    svm_img_tab = train_and_eval_svm(
        X_img_plus_tab,
        y_img_all,
        name="SVM (image + all tabular)",
    )
    cross_validation_svm(X_img_plus_tab, y_img_all, "SVM (image + all tabular)")

    # ------------------------
    # 5) SVM on images + first-order features
    # ------------------------
    X_images_f = ds_img_first["images"]       # (N, H, W)
    X_first2   = ds_img_first["tabular"]      # (N, 5)
    y_img_f    = ds_img_first["y"]

    Nf, Hf, Wf = X_images_f.shape
    X_img_flat_f = X_images_f.reshape(Nf, Hf * Wf)
    X_img_plus_first = np.concatenate([X_img_flat_f, X_first2], axis=1)

    svm_img_first = train_and_eval_svm(
        X_img_plus_first,
        y_img_f,
        name="SVM (image + first-order)",
    )
    cross_validation_svm(X_img_plus_first, y_img_f, "SVM (image + first-order)")

    # ------------------------
    # 6) SVM on images + second-order features
    # ------------------------
    X_images_s = ds_img_second["images"]      # (N, H, W)
    X_second2  = ds_img_second["tabular"]     # (N, 8)
    y_img_s    = ds_img_second["y"]

    Ns, Hs, Ws = X_images_s.shape
    X_img_flat_s = X_images_s.reshape(Ns, Hs * Ws)
    X_img_plus_second = np.concatenate([X_img_flat_s, X_second2], axis=1)

    svm_img_second = train_and_eval_svm(
        X_img_plus_second,
        y_img_s,
        name="SVM (image + second-order)",
    )
    cross_validation_svm(X_img_plus_second, y_img_s, "SVM (image + second-order)")

    # ------------------------
    # 7) SVM on images only
    # ------------------------
    X_images_only = ds_img_only["images"]     # (N, H, W)
    y_img_only    = ds_img_only["y"]

    No, Ho, Wo = X_images_only.shape
    X_img_only_flat = X_images_only.reshape(No, Ho * Wo)

    svm_img_only = train_and_eval_svm(
        X_img_only_flat,
        y_img_only,
        name="SVM (image only)",
    )
    cross_validation_svm(X_img_only_flat, y_img_only, "SVM (image only)")


    

    # ------------------------
    # Random-label sanity checks
    # ------------------------
    print("\n=== RANDOM LABEL SANITY CHECKS ===")
    random_label_sanity(ds_tab["X"], ds_tab["y"], "tabular-only")
    random_label_sanity(ds_first["X"], ds_first["y"], "first-order")
    random_label_sanity(ds_second["X"], ds_second["y"], "second-order")
    random_label_sanity(X_img_plus_tab, y_img_all, "image + all tabular")
    random_label_sanity(X_img_only_flat, y_img_only, "image only")