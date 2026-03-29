import numpy as np
from google.colab import drive
drive.mount('/content/drive')

esm2 = np.load(f"/content/drive/MyDrive/Senior Year/Undergraduate Research Program/cafa-6-protein-function-prediction/Train/train_esm2_embeddings.npy")
prott5 = np.load(f"/content/drive/MyDrive/Senior Year/Undergraduate Research Program/cafa-6-protein-function-prediction/Train/train_prott5_embeddings.npy")
ids = np.load(f"/content/drive/MyDrive/Senior Year/Undergraduate Research Program/cafa-6-protein-function-prediction/Train/train_protein_ids.npy", allow_pickle=True)

print(f"ESM-2:  {esm2.shape}")
print(f"ProtT5: {prott5.shape}")
print(f"IDs:    {ids.shape}")
print(f"First 5 IDs: {ids[:5]}")

assert esm2.shape == (82404, 1280), f"ESM-2 shape wrong: {esm2.shape}"
assert prott5.shape == (82404, 1024), f"ProtT5 shape wrong: {prott5.shape}"
assert ids.shape[0] == 82404, f"IDs count wrong: {ids.shape[0]}"
print("\nAll shapes correct.")

concat = np.concatenate([esm2, prott5], axis=1)
print(f"Concatenated: {concat.shape}")

np.save(f"/content/drive/MyDrive/Senior Year/Undergraduate Research Program/cafa-6-protein-function-prediction/Train/train_concat_embeddings.npy", concat)
print("Saved.")

import pandas as pd

terms = pd.read_csv(f"{DATA_DIR}/train_terms.tsv", sep="\t")
print(f"Raw: {len(terms)} rows, {terms['term'].nunique()} GO terms")

# Filter to GO terms with >= 50 proteins
freq = terms['term'].value_counts()
filtered = terms[terms['term'].isin(freq[freq >= 50].index)]
print(f"Filtered: {len(filtered)} rows, {filtered['term'].nunique()} terms")

# Get valid protein IDs
valid = filtered['EntryID'].unique()
print(f"Valid proteins: {len(valid)}")

# Index into embeddings
id_to_idx = {p: i for i, p in enumerate(ids)}
keep_idx = np.array([id_to_idx[p] for p in valid if p in id_to_idx])
keep_ids = np.array([p for p in valid if p in id_to_idx])
print(f"Matched: {len(keep_idx)}")

esm2_f = esm2[keep_idx]
prott5_f = prott5[keep_idx]
concat_f = concat[keep_idx]

print(f"ESM-2: {esm2_f.shape}, ProtT5: {prott5_f.shape}, Concat: {concat_f.shape}")

np.save(f"{DATA_DIR}/filtered_esm2.npy", esm2_f)
np.save(f"{DATA_DIR}/filtered_prott5.npy", prott5_f)
np.save(f"{DATA_DIR}/filtered_concat.npy", concat_f)
np.save(f"{DATA_DIR}/filtered_protein_ids.npy", keep_ids)
filtered.to_csv(f"{DATA_DIR}/filtered_terms.tsv", sep="\t", index=False)
print("Saved.")

