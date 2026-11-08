"""Episode-centric multiple-instance model: shared episode encoder, speed-grid encoder, window adapter, pooling, heads.

Instances of a drive bag are 16-s episode crops (M, 160, 22) with a type id and a 9-float context, optionally a
speed-indexed grid (ACC / DEC episodes) fused into the episode embedding (h += has_grid * Linear(h_grid)), and 60-s
window embeddings of an existing window model (instance type 8) through a small adapter. DriveMIL pools the instance
embeddings of a bag with gated attention (Ilse et al. 2018, one attention branch per task) or with a reliability-
weighted mean of the instance logits (mean_logit: g_i = softplus(MLP(ctx_i (+) type_emb_i))), and predicts the size
class and BEV-vs-ICE plus the auxiliary targets (four-class powertrain, log footprint, log curb mass, two cumulative
size logits). Padded instance slots never reach the outputs (masked pooling). Pure PyTorch, no custom kernels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.mil_data import CROP_CHANNELS, CTX_DIM, GRID_CHANNELS, N_TYPES, WINDOW_TYPE_ID
from experiments.models import ConvStem, ResBlock1d, n_params  # noqa: F401  (n_params re-exported for the trainer)

POOLS = ("gated_attn", "mean_logit")
HEADS = ("size", "bev", "pt4", "footprint", "logmass", "ordinal")
BRANCH_OF = {"size": 0, "footprint": 0, "logmass": 0, "ordinal": 0, "bev": 1, "pt4": 1}   # attention branch per head
N_BRANCHES = 2


# ----------------------------------------------------------------------------- instance encoders
class EpisodeEncoder(nn.Module):
    """(M, 160, 22) crop + type id + context -> (M, d): ConvStem (stride 4) -> 2 x ResBlock1d -> mean/max pool ->
    Linear(2d, d) + type embedding + context MLP -> LayerNorm."""

    def __init__(self, in_ch: int = len(CROP_CHANNELS), d: int = 128, n_types: int = N_TYPES, ctx_dim: int = CTX_DIM, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.stem = ConvStem(in_ch, d, dropout)
        self.blocks = nn.Sequential(ResBlock1d(d, d, 1, dropout), ResBlock1d(d, d, 1, dropout))
        self.proj = nn.Linear(2 * d, d)
        self.type_emb = nn.Embedding(n_types, d)
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, type_id: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((0, self.d), dtype=torch.float32)
        h = self.stem(x.float())                                   # (M, L/4, d)
        h = self.blocks(h.transpose(1, 2)).transpose(1, 2)         # (M, L/4, d)
        z = self.proj(torch.cat([h.mean(1), h.amax(1)], dim=-1))
        return self.norm(z + self.type_emb(type_id) + self.ctx_mlp(ctx.float()))


class SpeedGridEncoder(nn.Module):
    """(M_g, 72, 6) speed-indexed grid -> (M_g, d): Conv1d k5 x 2 + mean/max pool + Linear."""

    def __init__(self, in_ch: int = len(GRID_CHANNELS), d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.c1 = nn.Conv1d(in_ch, d // 2, 5, padding=2)
        self.c2 = nn.Conv1d(d // 2, d, 5, padding=2)
        self.proj = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        if g.shape[0] == 0:
            return g.new_zeros((0, self.d), dtype=torch.float32)
        h = F.gelu(self.c1(g.float().transpose(1, 2)))
        h = F.gelu(self.c2(h))
        return self.drop(self.proj(torch.cat([h.mean(-1), h.amax(-1)], dim=-1)))


class WindowAdapter(nn.Module):
    """(M_w, dim_in) window embedding -> (M_w, d) instance of type WINDOW_TYPE_ID (the type vector is added here)."""

    def __init__(self, dim_in: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, e: torch.Tensor, type_vec: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(e.float()) + type_vec.float())


# ----------------------------------------------------------------------------- pooling
def masked_softmax(s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """softmax over the instance axis of s (B, K, n) restricted to mask (B, K); padded slots get 0, an empty bag all 0."""
    m = mask.unsqueeze(-1)
    a = torch.softmax(s.float().masked_fill(~m, -1e4), dim=1) * m.float()
    return a / a.sum(1, keepdim=True).clamp_min(1e-12)


def masked_normalise(g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Non-negative weights g (B, K, n) -> weights that sum to 1 over the valid instances (0 on padded slots)."""
    gm = g.float() * mask.unsqueeze(-1).float()
    return gm / gm.sum(1, keepdim=True).clamp_min(1e-12)


class GatedAttentionPool(nn.Module):
    """Gated attention (Ilse et al. 2018): a_k = w^T (tanh(V h_k) * sigmoid(U h_k)), masked softmax over the instances,
    one attention vector w per branch. (B, K, d), mask (B, K) -> pooled (B, n_branches, d), attention (B, K, n_branches)."""

    def __init__(self, d: int, n_branches: int = N_BRANCHES, d_attn: int | None = None):
        super().__init__()
        d_attn = d_attn or d
        self.V = nn.Linear(d, d_attn)
        self.U = nn.Linear(d, d_attn)
        self.w = nn.Linear(d_attn, n_branches)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = masked_softmax(s, mask)
        z = torch.einsum("bkn,bkd->bnd", a, h.float())
        return z, a


class ReliabilityWeight(nn.Module):
    """g_i = softplus(MLP(ctx_i (+) type_emb_i)) per branch: the learned reliability of an instance (mean_logit pooling)."""

    def __init__(self, ctx_dim: int = CTX_DIM, n_types: int = N_TYPES, d_type: int = 16, n_branches: int = N_BRANCHES, hidden: int = 64):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, d_type)
        self.net = nn.Sequential(nn.Linear(ctx_dim + d_type, hidden), nn.GELU(), nn.Linear(hidden, n_branches))

    def forward(self, ctx: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(torch.cat([ctx.float(), self.type_emb(type_id).float()], dim=-1)))


def bag_head(d: int, out: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, out))


# ----------------------------------------------------------------------------- the drive model
class DriveMIL(nn.Module):
    """Drive bag -> size / bev logits + auxiliary heads + instance logits + attention.

    pool = "gated_attn": instance embeddings pooled per branch (0: size, footprint, logmass, ordinal; 1: bev, pt4), then
    bag MLP heads. pool = "mean_logit": every head is an instance-level linear layer and the bag output is the
    reliability-weighted mean of the instance logits (sum_i g_i l_i / sum_i g_i) with the branch's g."""

    def __init__(self, d: int = 128, pool: str = "gated_attn", use_grid: bool = True, use_windows: bool = True, wemb_dim: int = 0,
                 n_size: int = 3, n_bev: int = 2, n_pt4: int = 4, in_ch: int = len(CROP_CHANNELS), ctx_dim: int = CTX_DIM,
                 n_types: int = N_TYPES, dropout: float = 0.1):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool!r}; choose from {POOLS}")
        self.d, self.pool_kind, self.ctx_dim = d, pool, ctx_dim
        self.enc = EpisodeEncoder(in_ch, d, n_types, ctx_dim, dropout)
        self.use_grid = bool(use_grid)
        self.grid_enc = SpeedGridEncoder(len(GRID_CHANNELS), d, dropout) if use_grid else None
        self.grid_fuse = nn.Linear(d, d) if use_grid else None
        self.use_windows = bool(use_windows and wemb_dim > 0)
        self.win_adapter = WindowAdapter(wemb_dim, d, dropout) if self.use_windows else None
        self.n_out = dict(size=n_size, bev=n_bev, pt4=n_pt4, footprint=1, logmass=1, ordinal=n_size - 1)
        self.inst_size = nn.Linear(d, n_size)
        self.inst_bev = nn.Linear(d, n_bev)
        if pool == "gated_attn":
            self.pool = GatedAttentionPool(d, N_BRANCHES)
            self.heads = nn.ModuleDict({k: bag_head(d, v, dropout) for k, v in self.n_out.items()})
        else:
            self.rel = ReliabilityWeight(ctx_dim, n_types, 16, N_BRANCHES)
            self.heads = nn.ModuleDict({k: nn.Linear(d, v) for k, v in self.n_out.items() if k not in ("size", "bev")})

    @property
    def n_classes(self) -> dict:
        return dict(size=self.n_out["size"], bev=self.n_out["bev"])

    # ------------------------------------------------------------------ instances
    def instance_embeddings(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat instance embeddings (episodes first, then window instances) with their type, context, bag and slot."""
        x, type_id, ctx = batch["x"], batch["type"], batch["ctx"]
        h = self.enc(x, type_id, ctx)                                                   # (M, d)
        if self.use_grid and x.shape[0] and batch["grid"].shape[0]:
            gm = batch["grid_map"]
            has = gm >= 0
            hg = self.grid_fuse(self.grid_enc(batch["grid"]))                           # (M_g, d) in grid-row order
            h = h.index_add(0, torch.nonzero(has).squeeze(1), hg[gm[has]].to(h.dtype))
        hs, ts, cs, bs, ps = [h], [type_id], [ctx.float()], [batch["bag_idx"]], [batch["pos"]]
        if self.use_windows and batch["wemb"].shape[0]:
            hw = self.win_adapter(batch["wemb"], self.enc.type_emb.weight[WINDOW_TYPE_ID])
            mw = hw.shape[0]
            hs.append(hw.to(h.dtype))
            ts.append(torch.full((mw,), WINDOW_TYPE_ID, dtype=type_id.dtype, device=h.device))
            cs.append(torch.zeros((mw, self.ctx_dim), dtype=torch.float32, device=h.device))
            bs.append(batch["wemb_bag_idx"])
            ps.append(batch["wemb_pos"])
        return torch.cat(hs), torch.cat(ts), torch.cat(cs), torch.cat(bs), torch.cat(ps)

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(h (B, Kmax, d) with zeros on padded slots, mask (B, Kmax))."""
        h_all, _, _, bag, pos = self.instance_embeddings(batch)
        mask = batch["mask"]
        h_pad = h_all.new_zeros((mask.shape[0], mask.shape[1], h_all.shape[-1]), dtype=torch.float32)
        h_pad[bag, pos] = h_all.float()
        return h_pad, mask

    def forward(self, batch: dict, amp: bool = False) -> dict:
        dev = batch["x"].device.type
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bool(amp and dev == "cuda")):
            h_all, type_all, ctx_all, bag, pos = self.instance_embeddings(batch)
        h_all = h_all.float()
        mask = batch["mask"]
        B, K = mask.shape
        inst_size, inst_bev = self.inst_size(h_all), self.inst_bev(h_all)
        out = dict(inst_size=inst_size, inst_bev=inst_bev, inst_bag_idx=bag, inst_pos=pos, inst_type=type_all)
        if self.pool_kind == "gated_attn":
            h_pad = h_all.new_zeros((B, K, h_all.shape[-1]))
            h_pad[bag, pos] = h_all
            z, a = self.pool(h_pad, mask)                                               # (B, 2, d), (B, K, 2)
            for k in HEADS:
                out[k] = self.heads[k](z[:, BRANCH_OF[k]])
        else:
            g = self.rel(ctx_all, type_all)                                             # (M_all, 2)
            g_pad = g.new_zeros((B, K, g.shape[-1]))
            g_pad[bag, pos] = g
            a = masked_normalise(g_pad, mask)
            logits = dict(size=inst_size, bev=inst_bev, **{k: self.heads[k](h_all) for k in ("pt4", "footprint", "logmass", "ordinal")})
            for k, l in logits.items():
                l_pad = l.new_zeros((B, K, l.shape[-1]))
                l_pad[bag, pos] = l
                out[k] = torch.einsum("bk,bko->bo", a[..., BRANCH_OF[k]], l_pad)
            out["inst_g"] = g
        out["footprint"] = out["footprint"].squeeze(-1)
        out["logmass"] = out["logmass"].squeeze(-1)
        out["attn"] = a
        return out
