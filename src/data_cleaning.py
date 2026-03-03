import pandas as pd
import numpy as np
import math

from rdkit import Chem
from rdkit.Chem import Descriptors

from deepchem.feat import MolGraphConvFeaturizer, RDKitDescriptors


# ----------------------------------------------------------
# 1) Helper: RDKit mol from SMILES
# ----------------------------------------------------------
def smiles_to_mol(smi: str):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol
    except Exception:
        return None


# ----------------------------------------------------------
# 2) Helper: descriptor feasibility check
#    Uses MaxAbsPartialCharge as a proxy; if it fails/NaN,
#    we drop the molecule.
# ----------------------------------------------------------
all_descriptors = {name: func for name, func in Descriptors.descList}


def is_descriptor_fail(mol) -> bool:
    if mol is None:
        return True
    try:
        val = all_descriptors["MaxAbsPartialCharge"](mol)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return True
        return False
    except Exception:
        return True


# ----------------------------------------------------------
# 3) Helper: can DeepChem featurize this SMILES as a graph?
# ----------------------------------------------------------
graph_featurizer = MolGraphConvFeaturizer(use_edges=True)


def can_featurize_graph(smi: str) -> bool:
    try:
        feats = graph_featurizer.featurize([smi])
        g0 = feats[0]
        if g0 is None:
            return False
        if isinstance(g0, np.ndarray) and g0.size == 0:
            return False
        return True
    except Exception:
        return False


# ----------------------------------------------------------
# 4) Main cleaning function
# ----------------------------------------------------------
def clean_dataset(
    input_path: str,
    output_path: str,
    smiles_col: str = "SMILES",
):
    """
    Clean a SMILES-based dataset by:
      1) Dropping invalid SMILES
      2) Dropping molecules failing a descriptor feasibility check
      3) Dropping molecules failing DeepChem graph featurization
      4) Dropping rows with NaN/Inf in RDKitDescriptors
      5) Dropping rows with NaN/Inf in any numeric column (e.g., targets)

    Writes cleaned CSV to output_path and returns the cleaned DataFrame.
    """
    df = pd.read_csv(input_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Input CSV must have a '{smiles_col}' column")

    df[smiles_col] = df[smiles_col].astype(str)

    n0 = len(df)
    print(f"[Step 0] Initial rows: {n0}")

    # ---------- Step 1: remove invalid SMILES ----------
    mols = [smiles_to_mol(s) for s in df[smiles_col]]
    valid_mask = np.array([m is not None for m in mols])
    if not valid_mask.all():
        print(f"[Step 1] Dropping {np.sum(~valid_mask)} rows with invalid SMILES.")
    df = df[valid_mask].reset_index(drop=True)
    mols = [m for m in mols if m is not None]

    # ---------- Step 2: remove descriptor failures ----------
    fail_mask = np.array([is_descriptor_fail(m) for m in mols])
    if fail_mask.any():
        print(f"[Step 2] Dropping {np.sum(fail_mask)} rows failing descriptor feasibility check.")
    df = df[~fail_mask].reset_index(drop=True)

    smiles_list = df[smiles_col].tolist()

    # ---------- Step 3: remove rows that cannot be graph-featurized ----------
    graph_ok_mask = np.array([can_featurize_graph(s) for s in smiles_list])
    if not graph_ok_mask.all():
        print(f"[Step 3] Dropping {np.sum(~graph_ok_mask)} rows failing DeepChem graph featurization.")
    df = df[graph_ok_mask].reset_index(drop=True)

    smiles_list = df[smiles_col].tolist()

    # ---------- Step 4: remove rows with NaN / Inf RDKit descriptors ----------
    print("[Step 4] Checking RDKitDescriptors for NaN/Inf...")
    rd_featurizer = RDKitDescriptors()
    rd_feats = rd_featurizer.featurize(smiles_list)
    rd_feats = np.asarray(rd_feats, dtype=float)

    nonfinite_rd_mask = ~np.isfinite(rd_feats).all(axis=1)
    if nonfinite_rd_mask.any():
        print(f"[Step 4] Dropping {np.sum(nonfinite_rd_mask)} rows with NaN/Inf in RDKit descriptors.")
    df = df[~nonfinite_rd_mask].reset_index(drop=True)

    # ---------- Step 5: drop any rows with NaN/Inf in numeric columns ----------
    print("[Step 5] Dropping rows with NaN/Inf in any numeric column...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        nan_mask = df[numeric_cols].isna().any(axis=1)
        numeric_values = df[numeric_cols].values
        nonfinite_mask = ~np.isfinite(numeric_values).all(axis=1)
        bad_numeric_mask = nan_mask | nonfinite_mask

        if bad_numeric_mask.any():
            print(f"[Step 5] Dropping {bad_numeric_mask.sum()} rows with NaN/Inf in numeric data.")
        df_clean = df[~bad_numeric_mask].reset_index(drop=True)
    else:
        print("[Step 5] No numeric columns found; skipping NaN/Inf numeric filter.")
        df_clean = df.reset_index(drop=True)

    n_final = len(df_clean)
    print(f"[Done] Final rows after all cleaning: {n_final} (dropped {n0 - n_final})")

    df_clean.to_csv(output_path, index=False)
    print(f"[Saved] Cleaned dataset written to: {output_path}")

    return df_clean


if __name__ == "__main__":
    # Example usage (edit paths to your dataset)
    clean_dataset(
        input_path="data/input_dataset.csv",
        output_path="data/cleaned_dataset.csv",
        smiles_col="SMILES",
    )
