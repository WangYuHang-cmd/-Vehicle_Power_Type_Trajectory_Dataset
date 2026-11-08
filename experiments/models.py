"""Sequence classifiers for (B, 600, C) windows: 1-D ResNet, bi-GRU, Transformer encoder, Mamba-2 (SSD) and Mamba-1 (S6).

All sequence models share a convolutional stem that turns the 10 Hz input into 2.5 Hz tokens (stride 4, 150 tokens
of width d_model); the ResNet works on the raw 10 Hz signal. The Mamba blocks are pure-PyTorch implementations of
the published algorithms (Dao & Gu 2024 "state space duality" chunked scan for Mamba-2; the selective scan of Gu & Dao
2023 for Mamba-1); no custom CUDA kernels are needed. Classification uses bidirectional blocks (forward and reversed
scans, summed), as is usual for non-causal sequence classification.

Every network shares one API (SequenceClassifier): `tokens(x)` -> (B, L, d) features, `embed(x)` -> z (the pooled
vector after Head.fc1), `forward(x)` -> (B, n_classes) logits (unchanged), `forward_all(x)` -> dict with `cls`, `emb`
and the optional extra heads (`coral` ordinal logits, `fp` / `mass` regressions, `pt4` four-class powertrain, `dom`
conditional domain logits through a gradient-reversal layer). `Head.norm` keeps receiving the pooled feature and
`Head.fc2` stays the classification layer, so saved checkpoints and forward hooks keep working.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- shared pieces
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class ConvStem(nn.Module):
    """(B, L, C) -> (B, L/4, d): two strided convolutions + GELU, then LayerNorm over the width."""

    def __init__(self, in_ch: int, d: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, d, kernel_size=7, stride=2, padding=3)
        self.c2 = nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.gelu(self.c1(x))
        x = F.gelu(self.c2(x))
        x = x.transpose(1, 2)
        return self.drop(self.norm(x))


def pool_tokens(h: torch.Tensor) -> torch.Tensor:
    """(B, L, d) -> (B, 2d): mean and max over the tokens."""
    return torch.cat([h.mean(1), h.amax(1)], dim=-1)


class Pool(nn.Module):
    """mean + max pooling over tokens -> LayerNorm -> fc -> GELU -> dropout -> z (B, d_out)."""

    def __init__(self, d: int, d_out: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d)
        self.fc = nn.Linear(2 * d, d_out or d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # h: (B, L, d)
        return self.drop(F.gelu(self.fc(self.norm(pool_tokens(h)))))


class Head(nn.Module):
    """mean + max pooling over tokens -> LayerNorm -> MLP -> logits; `embed` stops before the last layer."""

    def __init__(self, d: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d)
        self.fc1 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d, n_classes)

    def embed(self, h: torch.Tensor) -> torch.Tensor:            # h: (B, L, d) -> z: (B, d)
        return self.drop(F.gelu(self.fc1(self.norm(pool_tokens(h)))))

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # h: (B, L, d)
        return self.fc2(self.embed(h))


class CoralHead(nn.Module):
    """CORAL ordinal head (Cao, Mirjalili & Raschka 2020): K-1 logits = w.z + b_j with one shared weight vector and
    biases ordered b_0 >= b_1 >= ... through a cumulative softplus, so P(y > j) = sigmoid(logit_j) is non-increasing in
    j for every input (rank consistency)."""

    def __init__(self, d: int, n_classes: int):
        super().__init__()
        if n_classes < 2:
            raise ValueError("CORAL needs at least two classes")
        self.n_classes = n_classes
        self.fc = nn.Linear(d, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(max(0, n_classes - 2)))

    def biases(self) -> torch.Tensor:
        steps = F.softplus(self.delta)
        return self.b0 - torch.cat([torch.zeros(1, device=steps.device, dtype=steps.dtype), torch.cumsum(steps, 0)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, d) -> (B, K-1)
        return self.fc(z) + self.biases()[None, :].to(z.dtype)


def coral_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(B,) class indices -> (B, K-1) binary targets t_j = 1[y > j]."""
    j = torch.arange(n_classes - 1, device=y.device)
    return (y[:, None] > j[None, :]).to(torch.float32)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """(B, K-1) CORAL logits -> (B, K) class probabilities P(y = j) = P(y > j-1) - P(y > j), clipped at 0 and
    renormalised (exact when the logits are rank-consistent)."""
    s = torch.sigmoid(logits.float())
    one = torch.ones_like(s[:, :1])
    zero = torch.zeros_like(s[:, :1])
    cum = torch.cat([one, s, zero], dim=1)                         # P(y > -1) = 1, ..., P(y > K-1) = 0
    p = (cum[:, :-1] - cum[:, 1:]).clamp_min(0.0)
    return p / p.sum(1, keepdim=True).clamp_min(1e-12)


class GradReverse(torch.autograd.Function):
    """Identity forward; the gradient is multiplied by -lambda on the way back."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lam)


class ConditionalDomainHead(nn.Module):
    """Conditional domain classifier (CDAN style): the reversed embedding is multiplied with the class posterior
    (outer product, d x K) and mapped to one logit per car model; training it with cross-entropy pushes the encoder
    towards class-conditional invariance across car models."""

    def __init__(self, d: int, n_models: int, n_classes: int, hidden: int | None = None):
        super().__init__()
        self.d, self.n_classes = d, n_classes
        hidden = hidden or d
        self.fc1 = nn.Linear(d * n_classes, hidden)
        self.fc2 = nn.Linear(hidden, n_models)

    def forward(self, z: torch.Tensor, cls_logits: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        f = grad_reverse(z, lam)
        g = torch.softmax(cls_logits.detach().float(), dim=-1).to(f.dtype)
        h = (f[:, :, None] * g[:, None, :]).reshape(z.shape[0], self.d * self.n_classes)
        return self.fc2(F.gelu(self.fc1(h)))


class MultiHead(nn.Module):
    """Named linear heads on the embedding z: {name: width}; the name "coral" builds a CoralHead whose width is the
    number of classes (it emits K-1 logits). forward(z) -> {name: output}."""

    def __init__(self, d: int, heads: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({k: (CoralHead(d, int(n)) if k == "coral" else nn.Linear(d, int(n))) for k, n in heads.items()})

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: m(z) for k, m in self.heads.items()}


class SequenceClassifier(nn.Module):
    """Shared API of the window classifiers (subclasses define the layers, `tokens(x)` and `self.head`)."""

    def _init_heads(self, d_z: int, n_classes: int, heads: dict[str, int] | None) -> None:
        self.n_classes = n_classes
        heads = dict(heads or {})
        n_dom = heads.pop("dom", None)
        self.heads = MultiHead(d_z, heads) if heads else None
        self.domain = ConditionalDomainHead(d_z, int(n_dom), n_classes) if n_dom else None

    def tokens(self, x: torch.Tensor) -> torch.Tensor:           # (B, L, C) -> (B, L', d)
        raise NotImplementedError

    def embed(self, x: torch.Tensor) -> torch.Tensor:            # (B, L, C) -> (B, d_z)
        return self.head.embed(self.tokens(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, L, C) -> (B, n_classes)
        return self.head.fc2(self.embed(x))

    def forward_all(self, x: torch.Tensor, grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        z = self.embed(x)
        out = {"cls": self.head.fc2(z), "emb": z}
        if self.heads is not None:
            out.update(self.heads(z))
        if self.domain is not None:
            out["dom"] = self.domain(z, out["cls"], grl_lambda)
        return out


# ----------------------------------------------------------------------------- 1-D ResNet on the raw signal
class ResBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, dropout: float):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.short = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x if self.short is None else self.short(x)
        x = F.gelu(self.b1(self.c1(x)))
        x = self.b2(self.c2(self.drop(x)))
        return F.gelu(x + s)


class ResNet1D(SequenceClassifier):
    def __init__(self, in_ch: int, n_classes: int, widths=(64, 128, 256), dropout: float = 0.1, heads: dict[str, int] | None = None):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, widths[0], 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks += [ResBlock1d(cin, w, stride=1 if i == 0 else 2, dropout=dropout), ResBlock1d(w, w, 1, dropout)]
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.head = Head(cin, n_classes, dropout)
        self._init_heads(cin, n_classes, heads)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x.transpose(1, 2))).transpose(1, 2)


# ----------------------------------------------------------------------------- GRU
class GRUNet(SequenceClassifier):
    def __init__(self, in_ch: int, n_classes: int, d: int = 128, n_layers: int = 2, dropout: float = 0.1,
                 heads: dict[str, int] | None = None):
        super().__init__()
        self.stem = ConvStem(in_ch, d, dropout)
        self.gru = nn.GRU(d, d, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = Head(2 * d, n_classes, dropout)
        self._init_heads(2 * d, n_classes, heads)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.stem(x))
        return h


# ----------------------------------------------------------------------------- Transformer encoder
class TransformerNet(SequenceClassifier):
    def __init__(self, in_ch: int, n_classes: int, d: int = 128, n_layers: int = 4, n_heads: int = 4, ff: int = 256,
                 dropout: float = 0.1, max_len: int = 256, heads: dict[str, int] | None = None):
        super().__init__()
        self.stem = ConvStem(in_ch, d, dropout)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = Head(d, n_classes, dropout)
        self._init_heads(d, n_classes, heads)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.pos[:, : h.shape[1]]
        return self.norm(self.enc(h))


# ----------------------------------------------------------------------------- Mamba-2 (SSD)
def segsum(x: torch.Tensor) -> torch.Tensor:
    """(..., T) -> (..., T, T) with out[..., i, j] = sum_{k=j+1..i} x[k] for i >= j, -inf above the diagonal."""
    T = x.shape[-1]
    xe = x.unsqueeze(-1).expand(*x.shape, T)
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    xe = xe.masked_fill(~strict, 0.0)
    xs = torch.cumsum(xe, dim=-2)
    incl = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return xs.masked_fill(~incl, float("-inf"))


def ssd_chunked(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Cm: torch.Tensor, block: int) -> torch.Tensor:
    """State-space duality scan (Dao & Gu 2024, Listing 1) without einops.

    X: (b, l, h, p) inputs already multiplied by dt; A: (b, l, h) = dt * A (<= 0); B, C: (b, l, h, n). Returns (b, l, h, p)."""
    b, l, h, p = X.shape
    n = B.shape[-1]
    c = l // block
    X = X.reshape(b, c, block, h, p)
    A = A.reshape(b, c, block, h).permute(0, 3, 1, 2)                          # (b, h, c, t)
    B = B.reshape(b, c, block, h, n)
    Cm = Cm.reshape(b, c, block, h, n)
    A_cumsum = torch.cumsum(A, dim=-1)                                            # (b, h, c, t)
    # 1. intra-chunk outputs (diagonal blocks)
    L = torch.exp(segsum(A))                                                      # (b, h, c, t, s)
    CB = torch.einsum("bclhn,bcshn->bclsh", Cm, B)                                # (b, c, t, s, h)
    Y_diag = torch.einsum("bclsh,bcshp->bclhp", CB * L.permute(0, 2, 3, 4, 1), X)
    # 2. per-chunk states
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)                      # (b, h, c, t)
    Bd = B * decay_states.permute(0, 2, 3, 1).unsqueeze(-1)                       # (b, c, t, h, n)
    states = torch.einsum("bclhn,bclhp->bchpn", Bd, X)                            # (b, c, h, p, n)
    # 3. inter-chunk recurrence
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)          # (b, c+1, h, p, n)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))             # (b, h, c+1, c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]                                                    # (b, c, h, p, n)
    # 4. state -> output
    Cs = Cm * torch.exp(A_cumsum).permute(0, 2, 3, 1).unsqueeze(-1)               # (b, c, t, h, n)
    Y_off = torch.einsum("bclhn,bchpn->bclhp", Cs, states)
    return (Y_diag + Y_off).reshape(b, l, h, p)


class Mamba2Block(nn.Module):
    """One Mamba-2 mixer (ngroups = 1): in_proj -> causal depthwise conv on (x, B, C) -> SSD -> gated RMSNorm -> out_proj."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64, chunk: int = 15):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state, self.headdim, self.chunk = d_state, headdim, chunk
        self.nheads = self.d_inner // headdim
        conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))           # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, _ = u.shape
        pad = (-l) % self.chunk
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, Cm = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)                                        # (b, l, h)
        A = -torch.exp(self.A_log.float())                                        # (h,)
        if pad:
            x = F.pad(x, (0, 0, 0, pad)); B = F.pad(B, (0, 0, 0, pad)); Cm = F.pad(Cm, (0, 0, 0, pad)); dt = F.pad(dt, (0, 0, 0, pad))
        L = l + pad
        xh = x.reshape(b, L, self.nheads, self.headdim)
        Bh = B.unsqueeze(2).expand(b, L, self.nheads, self.d_state)
        Ch = Cm.unsqueeze(2).expand(b, L, self.nheads, self.d_state)
        with torch.autocast(device_type=u.device.type, enabled=False):          # the scan runs in float32
            y = ssd_chunked((xh * dt.unsqueeze(-1)).float(), (dt * A).float(), Bh.float(), Ch.float(), self.chunk)
            y = y + xh.float() * self.D.view(1, 1, -1, 1)
        y = y.reshape(b, L, self.d_inner)[:, :l].to(u.dtype)
        y = self.norm(y * F.silu(z))
        return self.out_proj(y)


# ----------------------------------------------------------------------------- Mamba-1 (S6 selective scan)
class Mamba1Block(nn.Module):
    """The original selective SSM (sequential scan over the tokens, vectorised over batch, channel and state)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :l].transpose(1, 2))
        dt, B, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).float()                                 # (b, l, d)
        A = -torch.exp(self.A_log.float())                                        # (d, n)
        with torch.autocast(device_type=u.device.type, enabled=False):          # the scan runs in float32
            xf = x.float()
            dA = torch.exp(dt.unsqueeze(-1) * A)                                  # (b, l, d, n)
            dBx = (dt * xf).unsqueeze(-1) * B.float().unsqueeze(2)                # (b, l, d, n)
            h = torch.zeros(b, self.d_inner, self.d_state, device=u.device, dtype=torch.float32)
            ys = []
            Cf = Cm.float()
            for t in range(l):
                h = dA[:, t] * h + dBx[:, t]
                ys.append((h * Cf[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=1) + xf * self.D
        y = y.to(u.dtype) * F.silu(z)
        return self.out_proj(y)


class BiSSMLayer(nn.Module):
    """Pre-norm residual layer with a forward and a reversed SSM block (outputs summed)."""

    def __init__(self, d_model: int, make_block, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fwd = make_block()
        self.bwd = make_block()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        return x + self.drop(y)


class MambaNet(SequenceClassifier):
    def __init__(self, in_ch: int, n_classes: int, d: int = 128, n_layers: int = 4, dropout: float = 0.1, version: int = 2,
                 d_state: int | None = None, heads: dict[str, int] | None = None):
        super().__init__()
        self.stem = ConvStem(in_ch, d, dropout)
        if version == 2:
            ds = d_state or 64
            make = lambda: Mamba2Block(d, d_state=ds, headdim=64, chunk=15)      # noqa: E731
        else:
            ds = d_state or 16
            make = lambda: Mamba1Block(d, d_state=ds)                            # noqa: E731
        self.layers = nn.ModuleList([BiSSMLayer(d, make, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = Head(d, n_classes, dropout)
        self._init_heads(d, n_classes, heads)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


# ----------------------------------------------------------------------------- factory
MODELS = ("resnet", "gru", "transformer", "mamba2", "mamba1")


HEAD_KEYS = ("coral", "fp", "mass", "pt4", "dom")


def build_model(name: str, in_ch: int, n_classes: int, d: int = 128, n_layers: int = 4, dropout: float = 0.1,
                heads: dict[str, int] | None = None) -> nn.Module:
    """A window classifier; `heads` = optional extra heads {name: width} on the embedding ("coral": number of classes,
    "fp" / "mass": 1, "pt4": 4, "dom": number of car models). Without heads the module is the plain classifier whose
    forward(x) gives (B, n_classes) logits."""
    if heads:
        bad = sorted(set(heads) - set(HEAD_KEYS))
        if bad:
            raise ValueError(f"unknown heads {bad}; choose from {HEAD_KEYS}")
    if name == "resnet":
        return ResNet1D(in_ch, n_classes, dropout=dropout, heads=heads)
    if name == "gru":
        return GRUNet(in_ch, n_classes, d=d, n_layers=2, dropout=dropout, heads=heads)
    if name == "transformer":
        return TransformerNet(in_ch, n_classes, d=d, n_layers=n_layers, dropout=dropout, heads=heads)
    if name == "mamba2":
        return MambaNet(in_ch, n_classes, d=d, n_layers=n_layers, dropout=dropout, version=2, heads=heads)
    if name == "mamba1":
        return MambaNet(in_ch, n_classes, d=d, n_layers=n_layers, dropout=dropout, version=1, heads=heads)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
