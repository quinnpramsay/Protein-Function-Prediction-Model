# Overview
The goal of this project is to build a competitive model that can accurately predict the function of a protein based on its amino acid sequence. How I am going to achieve that is first using two embedding packages called ESM-2 and Prot-T5, as well as a concatenated version (to find out more about them, check the embedding section of the article). Then I am going to run them through two different types of models, XGBoost and a multilayer perceptron neural network, to hopefully create the best prediction possible. Not only will I use these prediction algorithms and embedding services to make the best possible outcome, but I will also use this as an opportunity to study the different embedding systems and the two models. At the end of this project, I will have 9 predictions and an in-depth analysis of their performance.

# Data Preprocessing
The data sourc efor this project is from the "Cafa 6 Protein Function Prediction" Kaggle Competition. There are three input files: train_sequences.fasta, train_terms.tsv, train_taxonomy.tsv. The raw data consists of 82,404 protein sequences, 537,027 annotation rows, and 26,125 unique GO terms across three ontologies (BP, MF, CC). Because there are so many sequences we removed ones where they contained non standard amino acids. Also I filtered GO terms to only those appearing in 50 or more proteins, reducing from 26,125 to 1,585 terms. Dropped ~6,000 proteins that lost all annotations after filtering, leaving 76,458 proteins. 
# Embedding

# XGBoost

# MLP Neural Network

# Predictions & Scoring

# Sources

https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data

https://peerj.com/articles/12019/
