from google.colab import drive
drive.mount('/content/drive')

import os, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

BASE_DIR = "/content/drive/MyDrive/Senior Year/Undergraduate Research Program"
DATA_DIR = os.path.join(BASE_DIR, "cafa-6-protein-function-prediction", "Train")

SEED = 67
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

terms_df = pd.read_csv(os.path.join(DATA_DIR, "filtered_terms.tsv"), sep="\t")
protein_ids = np.load(os.path.join(DATA_DIR, "filtered_protein_ids.npy"), allow_pickle=True)

sorted_terms = sorted(terms_df['term'].unique().tolist())
unsorted_terms = terms_df['term'].unique().tolist()
term_to_col = {t: i for i, t in enumerate(sorted_terms)}
num_terms = len(sorted_terms)

term_aspect = terms_df.drop_duplicates('term').set_index('term')['aspect'].to_dict()
bp_cols = [i for i, t in enumerate(sorted_terms) if term_aspect.get(t) == 'P']
mf_cols = [i for i, t in enumerate(sorted_terms) if term_aspect.get(t) == 'F']
cc_cols = [i for i, t in enumerate(sorted_terms) if term_aspect.get(t) == 'C']

pid_to_row = {pid: i for i, pid in enumerate(protein_ids)}
Y = np.zeros((len(protein_ids), num_terms), dtype=np.float32)
for _, row in terms_df.iterrows():
    pid, term = row['EntryID'], row['term']
    if pid in pid_to_row:
        Y[pid_to_row[pid], term_to_col[term]] = 1.0

unsorted_to_idx = {t: i for i, t in enumerate(unsorted_terms)}
xgb_reorder = [unsorted_to_idx[t] for t in sorted_terms]

dummy_X = np.zeros((len(protein_ids), 1))
_, _, _, Y_val = train_test_split(dummy_X, Y, test_size=0.2, random_state=SEED)
del dummy_X, Y
print(f"Val set: {Y_val.shape[0]} proteins")

class ProteinMLP(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mlp_val_preds(embed_file, model_file):
    embeddings = np.load(os.path.join(DATA_DIR, embed_file)).astype(np.float32)
    _, X_val_emb = train_test_split(embeddings, test_size=0.2, random_state=SEED)
    del embeddings
    checkpoint = torch.load(os.path.join(BASE_DIR, model_file), map_location="cpu", weights_only=False)
    model = ProteinMLP(checkpoint['input_dim'], checkpoint['num_terms']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_val_emb)), batch_size=1024, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(preds)

print("Generating MLP predictions...")
mlp_esm2 = get_mlp_val_preds("filtered_esm2.npy", "mlp_esm2_best.pt")
mlp_prott5 = get_mlp_val_preds("filtered_prott5.npy", "mlp_prott5_best.pt")
mlp_concat = get_mlp_val_preds("filtered_concat.npy", "mlp_concat.pt")
print("Done.")

xgb_esm2 = np.load(os.path.join(DATA_DIR, "filtered_esm2_xgb_preds.npy"))[:, xgb_reorder]
xgb_prott5 = np.load(os.path.join(DATA_DIR, "filtered_prott5_xgb_preds.npy"))[:, xgb_reorder]
xgb_concat = np.load(os.path.join(DATA_DIR, "filtered_concat_xgb_preds.npy"))[:, xgb_reorder]
print("XGBoost predictions loaded and reordered.")

ens_esm2 = (mlp_esm2 + xgb_esm2) / 2
ens_prott5 = (mlp_prott5 + xgb_prott5) / 2
ens_concat = (mlp_concat + xgb_concat) / 2

def compute_fmax(y_true, y_pred, indices=None):
    if indices is not None:
        y_true, y_pred = y_true[:, indices], y_pred[:, indices]
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

print("\n" + "=" * 65)
print("ENSEMBLE RESULTS (MLP + XGBoost averaged)")
print("=" * 65)

for name, preds in [("ESM-2", ens_esm2), ("ProtT5", ens_prott5), ("Concat", ens_concat)]:
    bp, _ = compute_fmax(Y_val, preds, bp_cols)
    mf, _ = compute_fmax(Y_val, preds, mf_cols)
    cc, _ = compute_fmax(Y_val, preds, cc_cols)
    cafa = (bp + mf + cc) / 3
    print(f"  {name:<10} BP={bp:.4f}  MF={mf:.4f}  CC={cc:.4f}  CAFA={cafa:.4f}")
