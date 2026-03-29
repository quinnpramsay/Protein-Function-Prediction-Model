# CELL 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# CELL 2 — Imports and Paths
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

BASE_DIR = "/content/drive/MyDrive/Senior Year/Undergraduate Research Program"
DATA_DIR = os.path.join(BASE_DIR, "cafa-6-protein-function-prediction", "Train")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# CELL 3 — Load Pre-Filtered Data
embeddings = np.load(os.path.join(DATA_DIR, "filtered_prott5.npy"))
protein_ids = np.load(os.path.join(DATA_DIR, "filtered_protein_ids.npy"), allow_pickle=True)
terms_df = pd.read_csv(os.path.join(DATA_DIR, "filtered_terms.tsv"), sep="\t")

print(f"Embeddings: {embeddings.shape}")
print(f"Protein IDs: {len(protein_ids)}")
print(f"Terms: {len(terms_df)} rows, {terms_df['term'].nunique()} unique GO terms")

assert embeddings.shape[0] == len(protein_ids), "Row mismatch between embeddings and protein IDs"

# CELL 4 — Build Label Matrix
frequent_terms = sorted(terms_df['term'].unique().tolist())
term_to_col = {t: i for i, t in enumerate(frequent_terms)}
num_terms = len(frequent_terms)

term_aspect = terms_df.drop_duplicates('term').set_index('term')['aspect'].to_dict()
bp_cols = [i for i, t in enumerate(frequent_terms) if term_aspect.get(t) == 'P']
mf_cols = [i for i, t in enumerate(frequent_terms) if term_aspect.get(t) == 'F']
cc_cols = [i for i, t in enumerate(frequent_terms) if term_aspect.get(t) == 'C']
print(f"{num_terms} terms — BP: {len(bp_cols)}, MF: {len(mf_cols)}, CC: {len(cc_cols)}")

pid_to_row = {pid: i for i, pid in enumerate(protein_ids)}
X = embeddings.astype(np.float32)
Y = np.zeros((len(protein_ids), num_terms), dtype=np.float32)

for _, row in terms_df.iterrows():
    pid, term = row['EntryID'], row['term']
    if pid in pid_to_row:
        Y[pid_to_row[pid], term_to_col[term]] = 1.0

print(f"X: {X.shape}, Y: {Y.shape}, label density: {Y.mean():.5f}")
del embeddings

# CELL 5 — Train/Val Split (80/20)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED)
print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}")
del X, Y

# CELL 6 — Model and Loss
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
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, x):
        return self.network(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()

# CELL 7 — Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BATCH_SIZE = 512
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 7

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
                        batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

model = ProteinMLP(X_train.shape[1], num_terms).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# CELL 8 — Train
best_val_loss = float('inf')
best_state = None
patience_counter = 0

print(f"{'Ep':>3} | {'Train':>10} | {'Val':>10} | {'LR':>9} | {'Time':>5} | Status")
print("-" * 60)

for epoch in range(MAX_EPOCHS):
    t0 = time.time()

    model.train()
    t_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * len(xb)
    t_loss /= len(train_loader.dataset)

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            v_loss += criterion(model(xb), yb).item() * len(xb)
    v_loss /= len(val_loader.dataset)

    scheduler.step(v_loss)
    elapsed = time.time() - t0

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        status = "✓ best"
    else:
        patience_counter += 1
        status = f"wait ({patience_counter}/{PATIENCE})"

    print(f"{epoch+1:>3} | {t_loss:>10.6f} | {v_loss:>10.6f} | {optimizer.param_groups[0]['lr']:>9.6f} | {elapsed:>4.1f}s | {status}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
model.to(device)
model.eval()
print(f"Best val loss: {best_val_loss:.6f}")

# CELL 9 — Predictions
all_preds = []
with torch.no_grad():
    for xb, _ in val_loader:
        probs = torch.sigmoid(model(xb.to(device)))
        all_preds.append(probs.cpu().numpy())
val_probs = np.concatenate(all_preds)
print(f"Predictions: {val_probs.shape}")

# CELL 10 — Fmax Evaluation
def compute_fmax(y_true, y_pred, indices=None):
    if indices is not None:
        y_true, y_pred = y_true[:, indices], y_pred[:, indices]
    has_label = y_true.sum(axis=1) > 0
    y_true, y_pred = y_true[has_label], y_pred[has_label]
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0

    best_f1, best_t, best_p, best_r = 0, 0, 0, 0
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
            best_f1, best_t, best_p, best_r = f1, t, prec, rec
    return best_f1, best_t, best_p, best_r

fmax_bp, t_bp, p_bp, r_bp = compute_fmax(Y_val, val_probs, bp_cols)
fmax_mf, t_mf, p_mf, r_mf = compute_fmax(Y_val, val_probs, mf_cols)
fmax_cc, t_cc, p_cc, r_cc = compute_fmax(Y_val, val_probs, cc_cols)
cafa_score = (fmax_bp + fmax_mf + fmax_cc) / 3

print(f"\n{'Ontology':<25} {'Fmax':>7} {'Thresh':>7} {'Prec':>7} {'Rec':>7}")
print("-" * 55)
print(f"{'Biological Process':<25} {fmax_bp:>7.4f} {t_bp:>7.2f} {p_bp:>7.4f} {r_bp:>7.4f}")
print(f"{'Molecular Function':<25} {fmax_mf:>7.4f} {t_mf:>7.2f} {p_mf:>7.4f} {r_mf:>7.4f}")
print(f"{'Cellular Component':<25} {fmax_cc:>7.4f} {t_cc:>7.2f} {p_cc:>7.4f} {r_cc:>7.4f}")
print("-" * 55)
print(f"{'CAFA Score':<25} {cafa_score:>7.4f}")

# CELL 11 — Save Model
SAVE_PATH = os.path.join(BASE_DIR, "mlp_prott5_best.pt")
torch.save({
    'model_state_dict': best_state,
    'num_terms': num_terms,
    'input_dim': X_train.shape[1],
    'best_val_loss': best_val_loss,
    'frequent_terms': frequent_terms,
    'term_aspect': term_aspect,
    'cafa_score': cafa_score,
    'fmax_bp': fmax_bp, 'fmax_mf': fmax_mf, 'fmax_cc': fmax_cc,
}, SAVE_PATH)
print(f"Saved to {SAVE_PATH}")
