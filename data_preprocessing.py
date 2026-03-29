import pandas as pd
import os
from google.colab import drive
# <<pip install BioPython>>
from Bio import SeqIO

drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/Senior Year/Undergraduate Research Program/cafa-6-protein-function-prediction"
os.listdir(DATA_DIR)

train_terms = pd.read_csv(f"{DATA_DIR}/Train/train_terms.tsv", sep="\t")

sequences = {r.id.split("|")[1]: str(r.seq) for r in SeqIO.parse(f"{DATA_DIR}/Train/train_sequences.fasta", "fasta")}

ia = pd.read_csv(f"{DATA_DIR}/IA.tsv", sep="\t", header=None, names=["term", "IA"])

term_counts = train_terms['term'].value_counts()
frequent_terms = term_counts[term_counts >= 50].index
filtered_terms = train_terms[train_terms['term'].isin(frequent_terms)]

seq_df = pd.DataFrame({'EntryID': list(sequences.keys()), 'sequence': list(sequences.values())})
seq_df.head()

import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

# Load ESM-2 model
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda").eval()

print("Model loaded!")

protein_ids = list(sequences.keys())
protein_seqs = list(sequences.values())

print(f"Generating embeddings for {len(protein_ids)} proteins...")

batch_size = 8
max_length = 1024
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(protein_seqs), batch_size)):
        batch_seqs = protein_seqs[i:i+batch_size]

        batch_seqs = [s[:max_length] for s in batch_seqs]

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length+2).to("cuda")

        outputs = model(**inputs)

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state
        masked = hidden * attention_mask
        embeddings = masked.sum(dim=1) / attention_mask.sum(dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

embeddings_matrix = np.concatenate(all_embeddings, axis=0)
print(f"Done! Shape: {embeddings_matrix.shape}")

np.save(f"{DATA_DIR}/embeddings_matrix.npy", embeddings_matrix)
np.save(f"{DATA_DIR}/protein_ids.npy", np.array(protein_ids))
