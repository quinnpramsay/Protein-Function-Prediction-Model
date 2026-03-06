"""
generate_esm2_embeddings.py
===========================
Generates ESM-2 embeddings for both train and test protein sequences.
Outputs:
    - train_esm2_embeddings.npy
    - train_protein_ids.npy
    - test_esm2_embeddings.npy
    - test_protein_ids.npy

Run on HPC with:
    srun --partition=gpus --nodelist=thor --gres=gpu:1 --time=480 \
        singularity run cafa.sif generate_esm2_embeddings.py

Or CPU-only:
    srun --partition=cs --nodelist=cnode1 --cpus-per-task=64 --time=720 \
        singularity run cafa.sif generate_esm2_embeddings.py
"""

import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ============================================================
# CONFIGURATION — update these paths to match your HPC layout
# ============================================================
DATA_DIR = os.path.expanduser("~/data/cafa-6-protein-function-prediction")
OUTPUT_DIR = os.path.expanduser("~/embeddings")
BATCH_SIZE = 4        # lower = less memory, increase if you have GPU
MAX_LENGTH = 1024     # ESM-2 max sequence length

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DETECT DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# LOAD ESM-2 MODEL
# ============================================================
print("Loading ESM-2 model...")
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()
print("Model loaded!")

# ============================================================
# FASTA PARSER (no BioPython needed)
# ============================================================
def parse_fasta(fasta_path):
    """Parse a FASTA file. Returns dict: {entry_id: sequence}"""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # Extract short ID from header like >sp|A0A0C5B5G6|MOTSC_HUMAN
                header = line[1:].split()[0]
                if "|" in header:
                    current_id = header.split("|")[1]
                else:
                    current_id = header
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences

# ============================================================
# EMBEDDING GENERATOR
# ============================================================
def generate_embeddings(sequences_dict, description=""):
    """
    Takes a dict of {protein_id: sequence} and returns
    embeddings matrix and protein_ids list.
    """
    protein_ids = list(sequences_dict.keys())
    protein_seqs = list(sequences_dict.values())

    all_embeddings = []

    print(f"Generating ESM-2 embeddings for {len(protein_ids)} {description} proteins...")

    with torch.no_grad():
        for i in tqdm(range(0, len(protein_seqs), BATCH_SIZE)):
            batch_seqs = [s[:MAX_LENGTH] for s in protein_seqs[i:i+BATCH_SIZE]]

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH + 2
            ).to(device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

            # Mean pooling (ignore padding tokens)
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    return embeddings_matrix, protein_ids

# ============================================================
# LOAD SEQUENCES
# ============================================================
print("\nLoading train sequences...")
train_seqs = parse_fasta(os.path.join(DATA_DIR, "Train", "train_sequences.fasta"))
print(f"  Loaded {len(train_seqs)} train sequences")

print("Loading test sequences...")
test_seqs = parse_fasta(os.path.join(DATA_DIR, "Test", "testsuperset.fasta"))
print(f"  Loaded {len(test_seqs)} test sequences")

# ============================================================
# GENERATE AND SAVE EMBEDDINGS
# ============================================================

# --- Train ---
train_embeddings, train_ids = generate_embeddings(train_seqs, "train")
np.save(os.path.join(OUTPUT_DIR, "train_esm2_embeddings.npy"), train_embeddings)
np.save(os.path.join(OUTPUT_DIR, "train_protein_ids.npy"), np.array(train_ids))
print(f"Saved train ESM-2 embeddings: {train_embeddings.shape}")

# --- Test ---
test_embeddings, test_ids = generate_embeddings(test_seqs, "test")
np.save(os.path.join(OUTPUT_DIR, "test_esm2_embeddings.npy"), test_embeddings)
np.save(os.path.join(OUTPUT_DIR, "test_protein_ids.npy"), np.array(test_ids))
print(f"Saved test ESM-2 embeddings: {test_embeddings.shape}")

print("\nDone! Files saved to:", OUTPUT_DIR)
print("  train_esm2_embeddings.npy")
print("  train_protein_ids.npy")
print("  test_esm2_embeddings.npy")
print("  test_protein_ids.npy")
