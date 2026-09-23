Drug-Target Binding Affinity Prediction
📋 Overview
This project performs exploratory data analysis (EDA) and feature extraction on the DAVIS dataset to study drug-protein interactions and predict binding affinity. The goal is to build machine learning models that can predict how strongly a drug compound will bind to a protein target.

📊 Dataset
DAVIS Kinase Binding Affinity Dataset
Source: DAVIS kinase binding affinity dataset
Size: Drug-protein interaction pairs with binding affinity values
Features:

💊 Drug molecules (SMILES format)
🧬 Protein targets (FASTA sequences)
📈 Binding affinity values (Y)


🔬 Features Extracted
Drug (SMILES) Features

Molecular weight - Total mass of the molecule
Number of atoms and bonds - Structural complexity
Number of rings - Total and aromatic ring structures
LogP - Lipophilicity (hydrophobicity)
TPSA - Topological polar surface area
Hydrogen bond donors/acceptors - Interaction potential

Protein (FASTA) Features

Sequence length - Number of amino acids
Molecular weight - Total protein mass
Isoelectric point - pH at neutral charge
Aromaticity - Aromatic amino acid content
Instability index - Protein stability measure
GRAVY score - Hydrophobicity index

🚀 Setup
```bash
conda env create -f environment.yml
conda activate dta
```

The dataset is fetched automatically via [PyTDC](https://tdcommons.ai/) rather than committed to the repo:
```bash
python -m src.features        # downloads DAVIS, extracts features -> data/davis_features.csv
python -m src.train_baseline  # trains baseline models -> results/
```

🤖 Baseline Models
Random Forest, XGBoost, and a Ridge regression baseline are trained on the extracted drug + protein descriptors to predict pKd (`-log10(Kd in M)`). Evaluated with RMSE, MAE, R², and Concordance Index (CI), the standard ranking metric for drug-target affinity tasks.

| Model | RMSE | MAE | R² | CI |
|---|---|---|---|---|
| Ridge | 0.790 | 0.538 | 0.114 | 0.667 |
| Random Forest | 0.652 | 0.399 | 0.396 | 0.815 |
| XGBoost | 0.645 | 0.404 | 0.408 | 0.816 |

Descriptor-based features top out around R²≈0.41 — the next step to push performance further is learned representations (e.g. molecular graph embeddings for drugs, CNN/transformer embeddings for protein sequences) instead of hand-crafted descriptors.
