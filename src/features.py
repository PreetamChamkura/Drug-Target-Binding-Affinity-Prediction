"""Feature extraction for SMILES (drug) and FASTA (protein) sequences.

Extracted from notebooks/davis_data_exploration.ipynb and extended to run
over the full dataset. Features are computed once per unique Drug/Target
entity (not per row) since DAVIS has many repeated drug-target pairs.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis

SMILES_FEATURE_NAMES = [
    "molecular_weight", "num_atoms", "num_bonds", "num_rings",
    "num_aromatic_rings", "logp", "tpsa", "hbd", "hba",
]

FASTA_FEATURE_NAMES = [
    "sequence_length", "molecular_weight", "isoelectric_point",
    "aromaticity", "instability_index", "gravy",
]


def extract_smiles_features(smiles):
    """Extract molecular features from a SMILES string."""
    empty = dict.fromkeys(SMILES_FEATURE_NAMES)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return empty
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
        }
    except Exception:
        return empty


def extract_fasta_features(fasta):
    """Extract sequence-derived features from a protein FASTA string."""
    empty = dict.fromkeys(FASTA_FEATURE_NAMES)
    try:
        if pd.isna(fasta) or fasta == "":
            return empty
        protein_analysis = ProteinAnalysis(fasta)
        return {
            "sequence_length": len(fasta),
            "molecular_weight": protein_analysis.molecular_weight(),
            "isoelectric_point": protein_analysis.isoelectric_point(),
            "aromaticity": protein_analysis.aromaticity(),
            "instability_index": protein_analysis.instability_index(),
            "gravy": protein_analysis.gravy(),
        }
    except Exception:
        return empty


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract drug + protein features for every row in df.

    df must have columns: Drug, Target, Y (Drug_ID/Target_ID optional).
    Features are computed once per unique SMILES/FASTA string and merged
    back onto the full row set for efficiency.
    """
    unique_drugs = df["Drug"].drop_duplicates()
    drug_features = pd.DataFrame(
        [extract_smiles_features(s) for s in unique_drugs]
    ).add_prefix("drug_")
    drug_features["Drug"] = unique_drugs.values

    unique_targets = df["Target"].drop_duplicates()
    target_features = pd.DataFrame(
        [extract_fasta_features(s) for s in unique_targets]
    ).add_prefix("target_")
    target_features["Target"] = unique_targets.values

    out = df.merge(drug_features, on="Drug", how="left")
    out = out.merge(target_features, on="Target", how="left")
    return out


def main():
    df = pd.read_csv("data/davis.csv")
    print(f"Loaded {len(df)} rows, {df['Drug'].nunique()} unique drugs, "
          f"{df['Target'].nunique()} unique targets")

    features_df = build_feature_table(df)

    feature_cols = [c for c in features_df.columns if c.startswith(("drug_", "target_"))]
    missing = features_df[feature_cols].isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values before drop:")
        print(missing)

    before = len(features_df)
    features_df = features_df.dropna(subset=feature_cols)
    print(f"Dropped {before - len(features_df)} rows with unparseable SMILES/FASTA")

    features_df.to_csv("data/davis_features.csv", index=False)
    print(f"Saved {len(features_df)} rows x {len(feature_cols)} features to data/davis_features.csv")


if __name__ == "__main__":
    main()
