"""
generate_prott5_embeddings.py
=============================
Generates ProtT5 (Rostlab/prot_t5_xl_uniref50) embeddings for protein sequences.
Each protein sequence is converted into a 1,024-dimensional vector using mean pooling.

Note: ProtT5 requires spaces between each amino acid character.
      "MKTLL" becomes "M K T L L" before tokenization.

Outputs:
    - train_prott5_embeddings.npy  (82,404 × 1,024)
    - test_prott5_embeddings.npy   (224,309 × 1,024)
    - train_protein_ids.npy
    - test_protein_ids.npy

Environment:
    - Singularity container (cafa.sif) with Python 3.12, PyTorch, transformers
    - Run on HPC CPU nodes via Slurm (no GPU required)
    - Models pre-downloaded to local cache (compute nodes have no internet)

Usage:
    sbatch --partition=cs --nodelist=cnode2 --cpus-per-task=64 --mem=256G \\
        --time=4320 --output=prott5_log.txt \\
        --wrap="singularity run ~/cafa.sif ~/generate_prott5_embeddings.py"
"""

import os
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.expanduser("~/data/cafa-6-protein-function-prediction")
OUTPUT_DIR = os.path.expanduser("~/embeddings")
MODEL_PATH = "/home/students/qramsay/models/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
BATCH_SIZE = 8
MAX_LENGTH = 1024      # Max sequence length
CHUNK_SIZE = 10000     # Save progress every 10,000 proteins

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading ProtT5 model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_PATH).to(device).eval()
print("Model loaded!")

# ============================================================
# FASTA PARSER
# ============================================================
def parse_fasta(fasta_path):
    """
    Parse a FASTA file into a dictionary of {entry_id: sequence}.
    Handles UniProt-style headers: >sp|SHORT_ID|NAME
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
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
# EMBEDDING GENERATOR WITH CHUNKED SAVING
# ============================================================
def generate_and_save(sequences_dict, prefix):
    """
    Generate ProtT5 embeddings and save in chunks.
    Resumes from last completed chunk if interrupted.

    Method: Mean pooling across sequence length, ignoring padding tokens.
    Each protein gets a single 1,024-dimensional vector.
    ProtT5 requires spaces between amino acids before tokenization.
    """
    protein_ids = list(sequences_dict.keys())
    protein_seqs = list(sequences_dict.values())

    # Save protein IDs
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_protein_ids.npy"), np.array(protein_ids))

    # Check which chunks are already done
    done_chunks = set()
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(f"{prefix}_prott5_chunk_") and f.endswith(".npy"):
            idx = int(f.replace(f"{prefix}_prott5_chunk_", "").replace(".npy", ""))
            done_chunks.add(idx)

    if done_chunks:
        print(f"  Resuming — already completed chunks: {sorted(done_chunks)}")

    print(f"Generating ProtT5 embeddings for {len(protein_ids)} {prefix} proteins...")

    with torch.no_grad():
        for chunk_start in range(0, len(protein_seqs), CHUNK_SIZE):
            chunk_idx = chunk_start // CHUNK_SIZE

            if chunk_idx in done_chunks:
                print(f"  Chunk {chunk_idx} already done, skipping")
                continue

            chunk_end = min(chunk_start + CHUNK_SIZE, len(protein_seqs))
            chunk_embeddings = []

            for i in tqdm(range(chunk_start, chunk_end, BATCH_SIZE),
                          desc=f"Chunk {chunk_idx}"):
                batch_seqs = protein_seqs[i:i+BATCH_SIZE]

                # ProtT5 requires spaces between each amino acid
                # "MKTLL" becomes "M K T L L"
                batch_seqs = [" ".join(list(s[:MAX_LENGTH])) for s in batch_seqs]

                inputs = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH + 2
                ).to(device)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                hidden = outputs.last_hidden_state

                # Mean pooling: average hidden states, ignoring padding
                mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                chunk_embeddings.append(embeddings.cpu().numpy())

            chunk_data = np.concatenate(chunk_embeddings, axis=0)
            np.save(os.path.join(OUTPUT_DIR, f"{prefix}_prott5_chunk_{chunk_idx}.npy"), chunk_data)
            print(f"  Saved chunk {chunk_idx}: {chunk_data.shape}")

    # Combine all chunks into one file
    print(f"Combining all {prefix} chunks...")
    chunks = sorted([f for f in os.listdir(OUTPUT_DIR)
                     if f.startswith(f"{prefix}_prott5_chunk_")])
    combined = np.concatenate([np.load(os.path.join(OUTPUT_DIR, c)) for c in chunks], axis=0)
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_prott5_embeddings.npy"), combined)
    print(f"  Final {prefix} ProtT5 embeddings: {combined.shape}")

    # Clean up chunks
    for c in chunks:
        os.remove(os.path.join(OUTPUT_DIR, c))

# ============================================================
# RUN
# ============================================================
print("\nLoading train sequences...")
train_seqs = parse_fasta(os.path.join(DATA_DIR, "Train", "train_sequences.fasta"))
print(f"  Loaded {len(train_seqs)} train sequences")

print("Loading test sequences...")
test_seqs = parse_fasta(os.path.join(DATA_DIR, "Test", "testsuperset.fasta"))
print(f"  Loaded {len(test_seqs)} test sequences")

generate_and_save(train_seqs, "train")
generate_and_save(test_seqs, "test")

print("\nDone! Files saved to:", OUTPUT_DIR)
