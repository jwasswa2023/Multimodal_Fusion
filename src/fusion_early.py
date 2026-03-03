# ==========================================
# CODE A
# Four-modality fusion:
#   Mol2Vec + RDKit + AttentiveFP STRICT embeddings + SMILES BiGRU embeddings
#   → LGBM + 3-seed ensemble
#   (collects ensemble_preds_fourmod & baseline_pred_fourmod for Code B)
# ==========================================
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl  # needed by AttentiveFP (DGL backend)

from rdkit import Chem

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer, RDKitDescriptors
from deepchem.data import NumpyDataset
from deepchem.models import AttentiveFPModel

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# =========================================================
# 0) Helpers
# =========================================================
def summarize(values, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mu = arr.mean()
    sd = arr.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        tcrit = 4.303 if n == 3 else 1.96
        margin = tcrit * sd / math.sqrt(n)
    else:
        margin = 0.0
    return mu, sd, mu - margin, mu + margin

def stratified_split_indices(y, test_size=0.2, n_bins=5, random_state=0):
    y = np.asarray(y).reshape(-1)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(y, qs))
    idx_all = np.arange(len(y))
    if len(bins) <= 2:
        print("[Warn] Stratification collapsed; using unstratified split.")
        y_binned = np.zeros_like(y, dtype=int)
    else:
        y_binned = np.digitize(y, bins[1:-1], right=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(idx_all, y_binned))
    return train_idx, test_idx

# ---------- STRICT AttentiveFP embedding extractor (DGL backend) ----------
def extract_attentivefp_embeddings_strict_dgl(model: AttentiveFPModel,
                                              dataset: dc.data.Dataset) -> np.ndarray:
    """
    Registers a pre-hook on the first nn.Linear in the FFN head to capture its input.
    If captured tensor is node-wise, mean-pool per graph using DGL batch_num_nodes().
    Returns (N_mols, d_embed).
    """
    assert hasattr(model, "model"), "Expected TorchModel with `.model` attribute."
    net = model.model          # inner nn.Module
    device = model.device

    # 1) Find a Linear layer in the FFN to hook
    target_linear = None
    if hasattr(net, "ffn") and isinstance(net.ffn, nn.Module):
        for sub in net.ffn.modules():
            if isinstance(sub, nn.Linear):
                target_linear = sub
                break
    if target_linear is None:
        for _name, sub in net.named_modules():
            if isinstance(sub, nn.Linear):
                target_linear = sub
                break
    if target_linear is None:
        raise RuntimeError("Could not find a Linear layer to hook in AttentiveFP.")

    captured = []
    def pre_hook(_mod, inputs):
        captured.append(inputs[0])

    handle = target_linear.register_forward_pre_hook(pre_hook)

    generator = model.default_generator(
        dataset,
        epochs=1,
        mode='predict',
        deterministic=True,
        pad_batches=False
    )

    pooled_chunks = []
    net.eval()
    with torch.no_grad():
        for batch in generator:
            # Prepare a DGL batch graph
            g, _labels, _weights = model._prepare_batch(batch)  # DGLGraph (batched)
            g = g.to(device)
            # Forward; hook captures the FFN input
            _ = net(g)
            if not captured:
                handle.remove()
                raise RuntimeError("Hook failed to capture tensor.")
            t = captured[-1]    # captured tensor for this forward

            # Pool to one vector per graph
            num_nodes_per_graph = g.batch_num_nodes().tolist()
            num_graphs = len(num_nodes_per_graph)
            total_nodes = sum(num_nodes_per_graph)

            if t.shape[0] == num_graphs:
                pooled = t  # already per-graph
            elif t.shape[0] == total_nodes:
                starts = np.cumsum([0] + num_nodes_per_graph[:-1])
                pooled_list = []
                for i, n in enumerate(num_nodes_per_graph):
                    s, e = starts[i], starts[i] + n
                    if n > 0:
                        pooled_list.append(t[s:e].mean(dim=0, keepdim=True))
                    else:
                        pooled_list.append(torch.zeros((1, t.shape[1]), device=t.device))
                pooled = torch.cat(pooled_list, dim=0)
            else:
                handle.remove()
                raise RuntimeError(
                    f"Cannot align captured tensor {tuple(t.shape)} with graphs: "
                    f"num_graphs={num_graphs}, total_nodes={total_nodes}."
                )
            pooled_chunks.append(pooled.cpu())
            captured.clear()

    handle.remove()
    if len(pooled_chunks) == 0:
        raise RuntimeError("No embeddings were collected.")
    emb = torch.cat(pooled_chunks, dim=0).numpy()
    return emb

# ---------- SMILES encoder (char-level BiGRU) ----------
class SmilesDataset(Dataset):
    def __init__(self, X_ids, y):
        self.X = torch.from_numpy(X_ids.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SmilesEncoderRegressor(nn.Module):
    """
    Returns a scalar prediction for training, but also stores the concatenated
    BiGRU hidden state (2*hidden_dim) in self.last_hidden for embedding extraction.
    """
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, pad_idx=0, p_drop=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(2*hidden_dim, 1)
        self.last_hidden = None

    def forward(self, x):
        emb = self.embedding(x)            # (B, L, E)
        out, h = self.gru(emb)             # h: (2, B, H)
        h_fw = h[0]
        h_bw = h[1]
        h_cat = torch.cat([h_fw, h_bw], dim=-1)  # (B, 2H)
        h_cat = self.dropout(h_cat)
        self.last_hidden = h_cat           # store for embedding extraction
        pred = self.fc(h_cat).squeeze(-1)  # (B,)
        return pred

def build_smiles_vocab(smiles_train, max_len_cap=200):
    special = ["<pad>", "<unk>"]
    chars = sorted({ch for s in smiles_train for ch in s})
    itos = special + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    max_len = min(max_len_cap, max(len(s) for s in smiles_train))
    return stoi, itos, max_len

def encode_smiles_array(smiles, stoi, max_len):
    pad_idx = stoi["<pad>"]; unk_idx = stoi["<unk>"]
    out = np.zeros((len(smiles), max_len), dtype=np.int64)
    for i, s in enumerate(smiles):
        ids = [stoi.get(ch, unk_idx) for ch in s[:max_len]]
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        out[i, :] = np.array(ids, dtype=np.int64)
    return out

@torch.no_grad()
def get_smiles_embeddings(model: SmilesEncoderRegressor, loader, device):
    model.eval()
    embs = []
    for xb, _ in loader:
        xb = xb.to(device)
        _ = model(xb)                 # forward fills model.last_hidden
        embs.append(model.last_hidden.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)  # (N, 2*hidden_dim)

# =========================================================
# 1) Load Mol2Vec + RDKit + targets + SMILES
# =========================================================
data = np.load("/content/mol2vec_rdkit_features.npz", allow_pickle=True)
X_m2v = data["X_mol2vec"]      # (N, d_m2v)
X_rd  = data["X_rdkit"]        # (N, d_rd)
y      = data["y"].astype(float)
smiles = data["smiles"].astype(str)

print("[Info] Loaded NPZ shapes:")
print("  Mol2Vec:", X_m2v.shape)
print("  RDKit  :", X_rd.shape)
print("  y      :", y.shape)
print("  SMILES :", smiles.shape)

N = len(y)
assert X_m2v.shape[0] == N and X_rd.shape[0] == N and smiles.shape[0] == N

# =========================================================
# 2) Build AttentiveFP graphs (DGL backend)
# =========================================================
graph_featurizer = MolGraphConvFeaturizer(use_edges=True)
X_graph = graph_featurizer.featurize(smiles.tolist())
mask = np.array([g is not None for g in X_graph])
if not mask.all():
    print(f"[Info] Dropping {np.sum(~mask)} molecules failing graph featurization.")

X_graph = np.asarray([g for g, ok in zip(X_graph, mask) if ok], dtype=object)
X_m2v   = X_m2v[mask]
X_rd    = X_rd[mask]
y       = y[mask]
smiles  = smiles[mask]

print("[Info] After graph alignment:")
print("  N       :", len(y))
print("  Mol2Vec :", X_m2v.shape)
print("  RDKit   :", X_rd.shape)
print("  Graph   :", X_graph.shape)

# =========================================================
# 3) Single stratified split (fixed)
# =========================================================
train_idx, test_idx = stratified_split_indices(y, test_size=0.2, n_bins=5, random_state=0)
print(f"[Info] Stratified split — Train: {len(train_idx)}, Test: {len(test_idx)}")

X_train_graph = X_graph[train_idx]
X_test_graph  = X_graph[test_idx]
X_train_m2v   = X_m2v[train_idx]
X_test_m2v    = X_m2v[test_idx]
X_train_rd    = X_rd[train_idx]
X_test_rd     = X_rd[test_idx]
y_train       = y[train_idx]
y_test        = y[test_idx]
smiles_train  = smiles[train_idx]
smiles_test   = smiles[test_idx]

train_dataset = NumpyDataset(X_train_graph, y_train.reshape(-1, 1))
test_dataset  = NumpyDataset(X_test_graph,  y_test.reshape(-1, 1))

# =========================================================
# 4) AttentiveFP (seed=0) → STRICT graph embeddings
# =========================================================
gnn_seed0 = AttentiveFPModel(
    n_tasks=1,
    mode="regression",
    batch_normalize=True,
    random_seed=0,
    model_dir="attfp_embed_seed0"
)
print("[Info] Training AttentiveFP (seed=0) for STRICT embedding extraction...")
gnn_seed0.fit(train_dataset, nb_epoch=50)

train_emb_attfp_seed0 = extract_attentivefp_embeddings_strict_dgl(gnn_seed0, train_dataset)
if train_emb_attfp_seed0.shape[0] != y_train.shape[0]:
    raise RuntimeError(f"AttentiveFP embedding count ({train_emb_attfp_seed0.shape[0]}) != train size ({y_train.shape[0]}).")
print("[Info] AttentiveFP embedding dim (seed=0):", train_emb_attfp_seed0.shape)

# =========================================================
# 5) SMILES sequence encoder (seed=0) → sequence embeddings
# =========================================================
stoi, itos, max_len = build_smiles_vocab(smiles_train, max_len_cap=200)
X_train_seq_ids = encode_smiles_array(smiles_train, stoi, max_len)
X_test_seq_ids  = encode_smiles_array(smiles_test,  stoi, max_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds_seq = SmilesDataset(X_train_seq_ids, y_train)
test_ds_seq  = SmilesDataset(X_test_seq_ids,  y_test)
train_loader_seq = DataLoader(train_ds_seq, batch_size=32, shuffle=True)
test_loader_seq  = DataLoader(test_ds_seq,  batch_size=64, shuffle=False)

torch.manual_seed(0)
seq_model = SmilesEncoderRegressor(vocab_size=len(itos), emb_dim=64, hidden_dim=128,
                                   pad_idx=stoi["<pad>"], p_drop=0.2).to(device)
criterion = nn.MSELoss()
optim = torch.optim.Adam(seq_model.parameters(), lr=1e-3)

print("[Info] Training SMILES sequence encoder (seed=0) for embeddings...")
for epoch in range(20):
    seq_model.train()
    running = 0.0
    for xb, yb in train_loader_seq:
        xb = xb.to(device); yb = yb.to(device)
        optim.zero_grad()
        preds = seq_model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optim.step()
        running += loss.item() * xb.size(0)

# Extract (B, 2H) embeddings via stored last_hidden
train_emb_smiles_seed0 = get_smiles_embeddings(seq_model, train_loader_seq, device)  # (n_train, 256)
print("[Info] SMILES embedding dim (seed=0):", train_emb_smiles_seed0.shape)

# =========================================================
# 6) Fuse FOUR representations for HP search on TRAIN:
#    [Mol2Vec || RDKit || AttentiveFP_emb || SMILES_emb]
# =========================================================
X_train_fused_seed0 = np.concatenate(
    [X_train_m2v, X_train_rd, train_emb_attfp_seed0, train_emb_smiles_seed0],
    axis=1
)

base_lgbm = LGBMRegressor()
param_distributions = {
    "verbose": [-1],
    "boosting_type": ["gbdt"],
    "num_leaves": [5, 15, 30],
    "max_depth": [50, 100, 300, -1],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample_for_bin": [50, 100, 200],
    "min_split_gain": [0.0],
    "min_child_weight": [0.001],
    "min_child_samples": [20],
    "subsample": [0.7, 0.8, 1.0],
    "feature_fraction": [0.7, 0.8, 1.0],
}
print("[Info] RandomizedSearchCV on FOUR-modality fused TRAIN features...")
rs = RandomizedSearchCV(
    estimator=base_lgbm,
    param_distributions=param_distributions,
    n_iter=60,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=0
)
rs.fit(X_train_fused_seed0, y_train)
best_params = rs.best_params_
print("[Info] Best LGBM params (four-modality):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# =========================================================
# 7) 3-seed eval:
#    retrain AttentiveFP & SMILES model per seed → get embeddings → fuse FOUR → LGBM
#    ALSO: collect ensemble predictions for Code B
# =========================================================
seeds = [0, 1, 2]
r2_list, rmse_list, mae_list = [], [], []

ensemble_preds_fourmod = []
baseline_pred_fourmod = None

print("\n" + "="*96)
print("Mol2Vec + RDKit + AttentiveFP (STRICT emb) + SMILES (BiGRU emb) + LGBM — 3-seed eval")
print("="*96)

for i, seed in enumerate(seeds):
    print(f"\n=== Seed {seed} ===")
    # AttentiveFP per-seed
    gnn = AttentiveFPModel(
        n_tasks=1,
        mode="regression",
        batch_normalize=True,
        random_seed=seed,
        model_dir=f"attfp_embed_seed{seed}"
    )
    gnn.fit(train_dataset, nb_epoch=50)
    train_emb_att = extract_attentivefp_embeddings_strict_dgl(gnn, train_dataset)
    test_emb_att  = extract_attentivefp_embeddings_strict_dgl(gnn, test_dataset)

    # SMILES BiGRU per-seed
    torch.manual_seed(seed)
    seq_model = SmilesEncoderRegressor(vocab_size=len(itos), emb_dim=64, hidden_dim=128,
                                       pad_idx=stoi["<pad>"], p_drop=0.2).to(device)
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

    # Align sanity
    if train_emb_att.shape[0] != y_train.shape[0] or test_emb_att.shape[0] != y_test.shape[0]:
        raise RuntimeError("AttentiveFP embedding count mismatch.")
    if train_emb_seq.shape[0] != y_train.shape[0] or test_emb_seq.shape[0] != y_test.shape[0]:
        raise RuntimeError("SMILES embedding count mismatch.")

    # Fuse FOUR modalities
    X_train_fused = np.concatenate([X_train_m2v, X_train_rd, train_emb_att, train_emb_seq], axis=1)
    X_test_fused  = np.concatenate([X_test_m2v,  X_test_rd,  test_emb_att,  test_emb_seq],  axis=1)

    params_seeded = best_params.copy()
    params_seeded["random_state"] = seed
    model = LGBMRegressor(**params_seeded)
    model.fit(X_train_fused, y_train)
    y_pred = model.predict(X_test_fused)

    # save for ensemble uncertainty
    ensemble_preds_fourmod.append(y_pred.copy())
    if i == 0:
        baseline_pred_fourmod = y_pred.copy()

    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R2   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    r2_list.append(r2); rmse_list.append(rmse); mae_list.append(mae)

# =========================================================
# 8) Summary
# =========================================================
r2_mean, r2_std, r2_lo, r2_hi = summarize(r2_list)
rmse_mean, rmse_std, rmse_lo, rmse_hi = summarize(rmse_list)
mae_mean, mae_std, mae_lo, mae_hi = summarize(mae_list)

print("\n=== Summary: Mol2Vec + RDKit + AttentiveFP-Emb + SMILES-Emb + LGBM ===")
print(f"R2   : {r2_mean:.4f} ± {r2_std:.4f}, 95% CI=({r2_lo:.4f}, {r2_hi:.4f})")
print(f"RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}, 95% CI=({rmse_lo:.4f}, {rmse_hi:.4f})")
print(f"MAE  : {mae_mean:.4f} ± {mae_std:.4f}, 95% CI=({mae_lo:.4f}, {mae_hi:.4f})")

# --- For Code B you now have:
#   y_test
#   ensemble_preds_fourmod
#   baseline_pred_fourmod
