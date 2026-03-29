import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/Senior Year/Undergraduate Research Program/cafa-6-protein-function-prediction/Train"

print("Loading data...")
embeddings = np.load(f"{DATA_DIR}/filtered_esm2.npy")
protein_ids = np.load(f"{DATA_DIR}/filtered_protein_ids.npy", allow_pickle=True).tolist()
filtered_terms = pd.read_csv(f"{DATA_DIR}/filtered_terms.tsv", sep="\t")

print(f"Embeddings: {embeddings.shape}")
print(f"Proteins: {len(protein_ids)}")
print(f"Term rows: {len(filtered_terms)}")

frequent_terms = filtered_terms["term"].unique().tolist()
print(f"GO terms: {len(frequent_terms)}")

term_aspect_map = filtered_terms.drop_duplicates("term").set_index("term")["aspect"]
term_aspects = [term_aspect_map[t] for t in frequent_terms]

print("Building label matrix...")
term_to_col = {t: i for i, t in enumerate(frequent_terms)}
pid_to_row = {p: i for i, p in enumerate(protein_ids)}
labels = np.zeros((len(protein_ids), len(frequent_terms)), dtype=np.float32)

valid = filtered_terms[
    filtered_terms["EntryID"].isin(pid_to_row) &
    filtered_terms["term"].isin(term_to_col)
]
row_idx = valid["EntryID"].map(pid_to_row).values
col_idx = valid["term"].map(term_to_col).values
labels[row_idx, col_idx] = 1.0

print(f"Label matrix: {labels.shape}")
print(f"Avg labels per protein: {labels.sum(axis=1).mean():.1f}")

X_train, X_val, y_train, y_val = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

print("Training XGBoost models...")
n_terms = labels.shape[1]
val_preds = np.zeros((X_val.shape[0], n_terms), dtype=np.float32)

for i in tqdm(range(n_terms), desc="XGBoost"):
    y_tr = y_train[:, i]
    y_vl = y_val[:, i]

    if y_tr.sum() == 0:
        continue

    n_neg = len(y_tr) - y_tr.sum()

    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.3,
        scale_pos_weight=n_neg / y_tr.sum(),
        eval_metric="logloss",
        verbosity=0,
        device="cuda",
        random_state=42,
        early_stopping_rounds=5,
    )
    model.fit(X_train, y_tr, eval_set=[(X_val, y_vl)], verbose=False)
    val_preds[:, i] = model.predict_proba(X_val)[:, 1]

print("Training complete.")


def compute_fmax(y_true, y_pred):
    has_label = y_true.sum(axis=1) > 0
    y_true, y_pred = y_true[has_label], y_pred[has_label]
    if len(y_true) == 0:
        return 0.0, 0.0

    best_f1, best_t = 0, 0
    for t in np.arange(0.01, 1.0, 0.02):
        pred_bin = (y_pred >= t).astype(float)
        if pred_bin.sum() == 0:
            continue
        tp = (pred_bin * y_true).sum(axis=1)
        n_pred = pred_bin.sum(axis=1)
        has_pred = n_pred > 0
        if has_pred.sum() == 0:
            continue
        prec = (tp[has_pred] / n_pred[has_pred]).mean()
        rec = (tp / y_true.sum(axis=1)).mean()
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


print("\nEvaluation:")
fmax, thresh = compute_fmax(y_val, val_preds)
print(f"Overall Fmax: {fmax:.4f} (threshold: {thresh:.3f})")

print("\nPer-aspect:")
term_aspects_arr = np.array(term_aspects)
aspect_fmax = {}

for aspect, code in [("BP", "P"), ("MF", "F"), ("CC", "C")]:
    mask = term_aspects_arr == code
    if mask.sum() == 0:
        continue
    fm, th = compute_fmax(y_val[:, mask], val_preds[:, mask])
    aspect_fmax[aspect] = fm
    print(f"  {aspect}: Fmax = {fm:.4f} (threshold = {th:.3f}, {mask.sum()} terms)")

cafa_score = np.mean(list(aspect_fmax.values()))
print(f"  CAFA Score: {cafa_score:.4f}")

save_path = f"{DATA_DIR}/xgboost_esm2_preds.npy"
np.save(save_path, val_preds)
print(f"\nSaved predictions to {save_path}")
