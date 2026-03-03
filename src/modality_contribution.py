# ============================================
# MODALITY CONTRIBUTION ANALYSIS (4 MODALITIES)
# Mol2Vec + RDKit + AttentiveFP_emb + SMILES_emb
# ============================================
import numpy as np
from typing import Dict, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import shap

# ------------------------------------------------
# 0. Helper functions
# ------------------------------------------------
def concat_blocks(blocks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, slice]]:
    """Horizontally stack blocks and remember each block's column slice."""
    keys, mats = zip(*blocks.items())
    sizes = [m.shape[1] for m in mats]
    X = np.hstack(mats)
    idx = {}
    st = 0
    for k, s in zip(keys, sizes):
        idx[k] = slice(st, st + s)
        st += s
    return X, idx

def summarize_scores(y_true, y_pred) -> dict:
    return dict(
        R2   = r2_score(y_true, y_pred),
        RMSE = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE  = float(mean_absolute_error(y_true, y_pred)),
    )

def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices (same samples)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kx = H @ Kx @ H
    Ky = H @ Ky @ H
    hsic = np.sum(Kx * Ky)
    norm = np.sqrt(np.sum(Kx*Kx) * np.sum(Ky*Ky)) + 1e-12
    return float(hsic / norm)

def delta(a: dict, b: dict) -> dict:
    """b - a for each metric key."""
    return {k: b[k] - a[k] for k in a}

def cca_corr(Xa, Xb, n_components=1):
    cca = CCA(n_components=n_components)
    U, V = cca.fit_transform(Xa, Xb)
    return float(np.corrcoef(U[:, 0], V[:, 0])[0, 1])

# ------------------------------------------------
# 1. Recompute embeddings for ONE reference seed (e.g. 0)
#    so we have clean, consistent representations for analysis
# ------------------------------------------------
ref_seed = 0
print(f"\n[Analysis] Recomputing embeddings for reference seed = {ref_seed}")

# --- AttentiveFP embeddings (graph) ---
gnn = AttentiveFPModel(
    n_tasks=1,
    mode="regression",
    batch_normalize=True,
    random_seed=ref_seed,
    model_dir=f"attfp_embed_seed{ref_seed}_analysis"
)
gnn.fit(train_dataset, nb_epoch=50)
train_emb_att = extract_attentivefp_embeddings_strict_dgl(gnn, train_dataset)
test_emb_att  = extract_attentivefp_embeddings_strict_dgl(gnn, test_dataset)

# --- SMILES BiGRU embeddings ---
torch.manual_seed(ref_seed)
seq_model = SmilesEncoderRegressor(
    vocab_size=len(itos),
    emb_dim=64,
    hidden_dim=128,
    pad_idx=stoi["<pad>"],
    p_drop=0.2
).to(device)
optim = torch.optim.Adam(seq_model.parameters(), lr=1e-3)

for epoch in range(20):
    seq_model.train()
    for xb, yb in train_loader_seq:
        xb = xb.to(device); yb = yb.to(device)
        optim.zero_grad()
        preds = seq_model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optim.step()

train_emb_seq = get_smiles_embeddings(seq_model, train_loader_seq, device)
test_emb_seq  = get_smiles_embeddings(seq_model, test_loader_seq,  device)

# --- Basic sanity ---
X_train_m2v = np.asarray(X_train_m2v, dtype=float)
X_test_m2v  = np.asarray(X_test_m2v,  dtype=float)
X_train_rd  = np.asarray(X_train_rd,  dtype=float)
X_test_rd   = np.asarray(X_test_rd,   dtype=float)
train_emb_att = np.asarray(train_emb_att, dtype=float)
test_emb_att  = np.asarray(test_emb_att,  dtype=float)
train_emb_seq = np.asarray(train_emb_seq, dtype=float)
test_emb_seq  = np.asarray(test_emb_seq,  dtype=float)

y_train = np.asarray(y_train, dtype=float).reshape(-1)
y_test  = np.asarray(y_test,  dtype=float).reshape(-1)

if train_emb_att.shape[0] != y_train.shape[0] or test_emb_att.shape[0] != y_test.shape[0]:
    raise RuntimeError("AttentiveFP emb count mismatch.")
if train_emb_seq.shape[0] != y_train.shape[0] or test_emb_seq.shape[0] != y_test.shape[0]:
    raise RuntimeError("SMILES emb count mismatch.")

print("[Analysis] Train shapes — Mol2Vec:", X_train_m2v.shape,
      "| RDKit:", X_train_rd.shape,
      "| GNN_emb:", train_emb_att.shape,
      "| SMILES_emb:", train_emb_seq.shape)

# Fuse for this reference seed
X_train_fused = np.hstack([X_train_m2v, X_train_rd, train_emb_att, train_emb_seq])
X_test_fused  = np.hstack([X_test_m2v,  X_test_rd,  test_emb_att,  test_emb_seq])

# Block dict (TRAIN) for analysis
blocks_train = {
    "mol2vec":   X_train_m2v,
    "rdkit":     X_train_rd,
    "gnn_emb":   train_emb_att,
    "smiles_emb": train_emb_seq,
}

# ------------------------------------------------
# 2. BEFORE MODELING — Mutual Information + CCA
# ------------------------------------------------
print("\n" + "="*100)
print("2. BEFORE MODELING — Complementarity (Mutual Information + CCA)")
print("="*100)

X_all_train, idx_blocks = concat_blocks(blocks_train)
mi_all = mutual_info_regression(
    X_all_train,
    y_train,
    random_state=0,
    n_neighbors=3
)

mi_m2v_med   = float(np.median(mi_all[idx_blocks["mol2vec"]]))
mi_rd_med    = float(np.median(mi_all[idx_blocks["rdkit"]]))
mi_gnn_med   = float(np.median(mi_all[idx_blocks["gnn_emb"]]))
mi_smiles_med = float(np.median(mi_all[idx_blocks["smiles_emb"]]))

print(f"[MI | train] Mol2Vec median   : {mi_m2v_med:.4f}")
print(f"[MI | train] RDKit median     : {mi_rd_med:.4f}")
print(f"[MI | train] GNN_emb median   : {mi_gnn_med:.4f}")
print(f"[MI | train] SMILES_emb median: {mi_smiles_med:.4f}")

# CCA between pairs (on standardized TRAIN)
A_m2v   = StandardScaler().fit_transform(X_train_m2v)
A_rd    = StandardScaler().fit_transform(X_train_rd)
A_gnn   = StandardScaler().fit_transform(train_emb_att)
A_smile = StandardScaler().fit_transform(train_emb_seq)

print("\n[CCA | train] Canonical correlations:")
print(f" Mol2Vec vs RDKit      : {cca_corr(A_m2v, A_rd):.4f}")
print(f" Mol2Vec vs GNN_emb    : {cca_corr(A_m2v, A_gnn):.4f}")
print(f" Mol2Vec vs SMILES_emb : {cca_corr(A_m2v, A_smile):.4f}")
print(f" RDKit   vs GNN_emb    : {cca_corr(A_rd,  A_gnn):.4f}")
print(f" RDKit   vs SMILES_emb : {cca_corr(A_rd,  A_smile):.4f}")
print(f" GNN_emb vs SMILES_emb : {cca_corr(A_gnn, A_smile):.4f}")

# ------------------------------------------------
# 3. BEFORE MODELING — Representation Geometry (Linear CKA)
# ------------------------------------------------
print("\n" + "="*100)
print("3. BEFORE MODELING — Representation Geometry (Linear CKA)")
print("="*100)

X_rep_m2v_tr   = StandardScaler().fit_transform(X_train_m2v)
X_rep_rd_tr    = StandardScaler().fit_transform(X_train_rd)
X_rep_gnn_tr   = StandardScaler().fit_transform(train_emb_att)
X_rep_smile_tr = StandardScaler().fit_transform(train_emb_seq)

print("[CKA | train] pairwise:")
print(f" Mol2Vec vs RDKit      : {linear_CKA(X_rep_m2v_tr, X_rep_rd_tr):.4f}")
print(f" Mol2Vec vs GNN_emb    : {linear_CKA(X_rep_m2v_tr, X_rep_gnn_tr):.4f}")
print(f" Mol2Vec vs SMILES_emb : {linear_CKA(X_rep_m2v_tr, X_rep_smile_tr):.4f}")
print(f" RDKit   vs GNN_emb    : {linear_CKA(X_rep_rd_tr,  X_rep_gnn_tr):.4f}")
print(f" RDKit   vs SMILES_emb : {linear_CKA(X_rep_rd_tr,  X_rep_smile_tr):.4f}")
print(f" GNN_emb vs SMILES_emb : {linear_CKA(X_rep_gnn_tr, X_rep_smile_tr):.4f}")

# RDKit vs fused on TEST
X_rep_rd_te    = StandardScaler().fit_transform(X_test_rd)
X_rep_fused_te = StandardScaler().fit_transform(X_test_fused)
cka_rd_fused_te = linear_CKA(X_rep_rd_te, X_rep_fused_te)
print(f"[CKA | test ] RDKit vs Fused representations: {cka_rd_fused_te:.4f}")

# ------------------------------------------------
# 4. AFTER MODELING — Fused vs 4 unimodal models
# ------------------------------------------------
print("\n" + "="*100)
print("4. AFTER MODELING — Fused vs Unimodal (Mol2Vec, RDKit, GNN_emb, SMILES_emb)")
print("="*100)

params0 = best_params.copy()
params0["random_state"] = ref_seed

fused_model = LGBMRegressor(**params0).fit(X_train_fused, y_train)
m2v_model   = LGBMRegressor(**params0).fit(X_train_m2v, y_train)
rd_model    = LGBMRegressor(**params0).fit(X_train_rd,  y_train)
gnn_model   = LGBMRegressor(**params0).fit(train_emb_att, y_train)
smile_model = LGBMRegressor(**params0).fit(train_emb_seq, y_train)

y_fused = fused_model.predict(X_test_fused)
y_m2v   = m2v_model.predict(X_test_m2v)
y_rd    = rd_model.predict(X_test_rd)
y_gnn   = gnn_model.predict(test_emb_att)
y_smile = smile_model.predict(test_emb_seq)

print("[Scores] Fused       :", summarize_scores(y_test, y_fused))
print("[Scores] Mol2Vec-only:", summarize_scores(y_test, y_m2v))
print("[Scores] RDKit-only  :", summarize_scores(y_test, y_rd))
print("[Scores] GNN_emb-only:", summarize_scores(y_test, y_gnn))
print("[Scores] SMILES-only :", summarize_scores(y_test, y_smile))

# ------------------------------------------------
# 5. AFTER MODELING — FI + SHAP (block-level)
# ------------------------------------------------
print("\n" + "="*100)
print("5. AFTER MODELING — Feature Importance + SHAP (grouped by modality)")
print("="*100)

fi = fused_model.feature_importances_  # length sum of all dims

fi_m2v   = float(np.sum(fi[idx_blocks["mol2vec"]]))
fi_rd    = float(np.sum(fi[idx_blocks["rdkit"]]))
fi_gnn   = float(np.sum(fi[idx_blocks["gnn_emb"]]))
fi_smile = float(np.sum(fi[idx_blocks["smiles_emb"]]))

print(f"[FI] Mol2Vec sum   : {fi_m2v:.1f}")
print(f"[FI] RDKit sum     : {fi_rd:.1f}")
print(f"[FI] GNN_emb sum   : {fi_gnn:.1f}")
print(f"[FI] SMILES_emb sum: {fi_smile:.1f}")

expl = shap.TreeExplainer(fused_model)
n_samp = min(1000, X_test_fused.shape[0])
rng = np.random.default_rng(0)
idx_samp = rng.choice(X_test_fused.shape[0], size=n_samp, replace=False)
X_te_s = X_test_fused[idx_samp]

sv = expl.shap_values(X_te_s)  # (n_samp, d_total)
sv_abs_mean = np.abs(sv).mean(axis=0)

shap_m2v   = float(np.sum(sv_abs_mean[idx_blocks["mol2vec"]]))
shap_rd    = float(np.sum(sv_abs_mean[idx_blocks["rdkit"]]))
shap_gnn   = float(np.sum(sv_abs_mean[idx_blocks["gnn_emb"]]))
shap_smile = float(np.sum(sv_abs_mean[idx_blocks["smiles_emb"]]))

print(f"[SHAP |mean|] Mol2Vec   : {shap_m2v:.4f}")
print(f"[SHAP |mean|] RDKit     : {shap_rd:.4f}")
print(f"[SHAP |mean|] GNN_emb   : {shap_gnn:.4f}")
print(f"[SHAP |mean|] SMILES_emb: {shap_smile:.4f}")
# shap.summary_plot(sv, X_te_s, plot_type="bar")  # optional

# ------------------------------------------------
# 6. AFTER MODELING — Ablation (drop-one-modality)
# ------------------------------------------------
print("\n" + "="*100)
print("6. AFTER MODELING — Ablation: Drop-one vs 4-way Fused")
print("="*100)

base_scores = summarize_scores(y_test, y_fused)

# Keep [RDKit, GNN_emb, SMILES_emb] -> drop Mol2Vec
X_train_drop_m2v = np.hstack([X_train_rd, train_emb_att, train_emb_seq])
X_test_drop_m2v  = np.hstack([X_test_rd,  test_emb_att,  test_emb_seq])
model_drop_m2v   = LGBMRegressor(**params0).fit(X_train_drop_m2v, y_train)
y_drop_m2v       = model_drop_m2v.predict(X_test_drop_m2v)
scores_drop_m2v  = summarize_scores(y_test, y_drop_m2v)

# Keep [Mol2Vec, GNN_emb, SMILES_emb] -> drop RDKit
X_train_drop_rd = np.hstack([X_train_m2v, train_emb_att, train_emb_seq])
X_test_drop_rd  = np.hstack([X_test_m2v,  test_emb_att,  test_emb_seq])
model_drop_rd   = LGBMRegressor(**params0).fit(X_train_drop_rd, y_train)
y_drop_rd       = model_drop_rd.predict(X_test_drop_rd)
scores_drop_rd  = summarize_scores(y_test, y_drop_rd)

# Keep [Mol2Vec, RDKit, SMILES_emb] -> drop GNN_emb
X_train_drop_gnn = np.hstack([X_train_m2v, X_train_rd, train_emb_seq])
X_test_drop_gnn  = np.hstack([X_test_m2v,  X_test_rd,  test_emb_seq])
model_drop_gnn   = LGBMRegressor(**params0).fit(X_train_drop_gnn, y_train)
y_drop_gnn       = model_drop_gnn.predict(X_test_drop_gnn)
scores_drop_gnn  = summarize_scores(y_test, y_drop_gnn)

# Keep [Mol2Vec, RDKit, GNN_emb] -> drop SMILES_emb
X_train_drop_smile = np.hstack([X_train_m2v, X_train_rd, train_emb_att])
X_test_drop_smile  = np.hstack([X_test_m2v,  X_test_rd,  test_emb_att])
model_drop_smile   = LGBMRegressor(**params0).fit(X_train_drop_smile, y_train)
y_drop_smile       = model_drop_smile.predict(X_test_drop_smile)
scores_drop_smile  = summarize_scores(y_test, y_drop_smile)

print("[Ablation] ALL (4-mod)        :", base_scores)
print("[Ablation] ALL - Mol2Vec      :", delta(base_scores, scores_drop_m2v))
print("[Ablation] ALL - RDKit        :", delta(base_scores, scores_drop_rd))
print("[Ablation] ALL - GNN_emb      :", delta(base_scores, scores_drop_gnn))
print("[Ablation] ALL - SMILES_emb   :", delta(base_scores, scores_drop_smile))
print("[Ablation] Mol2Vec-only       :", summarize_scores(y_test, y_m2v))
print("[Ablation] RDKit-only         :", summarize_scores(y_test, y_rd))
print("[Ablation] GNN_emb-only       :", summarize_scores(y_test, y_gnn))
print("[Ablation] SMILES_emb-only    :", summarize_scores(y_test, y_smile))

# ------------------------------------------------
# 7. DURING PREDICTION — Error correlation & fusion synergy
# ------------------------------------------------
print("\n" + "="*100)
print("7. DURING PREDICTION — Error Correlation & Fraction Improved")
print("="*100)

err_m2v   = y_test - y_m2v
err_rd    = y_test - y_rd
err_gnn   = y_test - y_gnn
err_smile = y_test - y_smile
err_fus   = y_test - y_fused

corr_m2v_fus   = np.corrcoef(err_m2v,   err_fus)[0, 1]
corr_rd_fus    = np.corrcoef(err_rd,    err_fus)[0, 1]
corr_gnn_fus   = np.corrcoef(err_gnn,   err_fus)[0, 1]
corr_smile_fus = np.corrcoef(err_smile, err_fus)[0, 1]

print(f"[Error corr] Mol2Vec vs Fused   : {corr_m2v_fus:.3f}")
print(f"[Error corr] RDKit   vs Fused   : {corr_rd_fus:.3f}")
print(f"[Error corr] GNN_emb vs Fused   : {corr_gnn_fus:.3f}")
print(f"[Error corr] SMILES_emb vs Fused: {corr_smile_fus:.3f}")

best_uni = np.minimum.reduce([
    np.abs(err_m2v),
    np.abs(err_rd),
    np.abs(err_gnn),
    np.abs(err_smile),
])
improve = best_uni - np.abs(err_fus)  # >0 ⇒ fused better than best unimodal

print(f"[Improvement] fraction of test samples improved by fusion: {(improve>0).mean():.2%}")





#PLOTTING CDF CURVES
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# SETTINGS
# =========================
SAVE_FIGS = True
FIG_DIR   = Path("./figs_modality_contribution")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CDF helper
# =========================
def _ecdf(x: np.ndarray):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs) if len(xs) > 0 else np.array([])
    return xs, ys

def plot_abs_error_cdf(
    y_true,
    preds_dict,
    title="CDF of absolute errors",
    save_path=None
):
    """
    y_true: array-like shape (N,)
    preds_dict: {"label": y_pred_array, ...}
    """
    y_true = np.asarray(y_true).reshape(-1)

    plt.figure(figsize=(5.5, 5.0))

    for label, y_pred in preds_dict.items():
        y_pred = np.asarray(y_pred).reshape(-1)
        if len(y_pred) != len(y_true):
            raise ValueError(f"Length mismatch for '{label}': {len(y_pred)} vs {len(y_true)}")

        abs_err = np.abs(y_true - y_pred)
        xs, ys = _ecdf(abs_err)
        plt.plot(xs, ys, label=label)

    plt.xlabel(r"|Error| = |y_true − y_pred|")
    plt.ylabel("Empirical CDF")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# CALL (edit labels as you like)
# =========================
preds = {
    "Mol2Vec-only": y_m2v,
    "RDKit-only": y_rd,
    "GNN_emb-only": y_gnn,
    "SMILES_emb-only": y_smile,
    "Fused (all modalities)": y_fused,
}

plot_abs_error_cdf(
    y_true=y_test,
    preds_dict=preds,
    title="Error CDF — unimodal vs fused",
    save_path=(FIG_DIR / "error_cdf_unimodal_vs_fused.png") if SAVE_FIGS else None
)
