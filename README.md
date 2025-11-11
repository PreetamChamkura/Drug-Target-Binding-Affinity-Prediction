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
