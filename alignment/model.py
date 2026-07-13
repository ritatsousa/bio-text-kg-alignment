"""Projector models: text embedding -> KG triple embedding space.

Available architectures
-----------------------
mlp             MLP with LayerNorm + GELU + Dropout blocks (original).
linear          Affine projection: Linear(input_dim, output_dim) with bias.
residual        MLP with additive skip connections between same-dim blocks.
bilinear        Pure matrix multiplication x @ W, no bias, no activation.
                Matches the left-factor of a bilinear scoring function
                score(text, triple) = (text @ W) · triple.
low_rank        Factorised W = A @ B with rank << min(input, output);
                rank = hidden_dims[0] (default 64).  No bias, no activation.
cross_attention 3 learnable queries (s, p, o intent) cross-attend over the
                text embedding.  d_model = hidden_dims[0] (default 256).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


# ---------------------------------------------------------------------------
# Architecture 1 — MLP (original)
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """MLP projector: [Linear -> LayerNorm -> GELU -> Dropout] x N + Linear."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend(_mlp_block(in_dim, h, dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture 2 — Linear
# ---------------------------------------------------------------------------

class LinearProjector(nn.Module):
    """Single linear projection: Linear(input_dim, output_dim).

    No hidden layers, no nonlinearity.  Serves as a strong, interpretable
    baseline to check whether nonlinear capacity is genuinely needed.
    ``hidden_dims`` is accepted but ignored so the factory interface is uniform.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],  # ignored
        output_dim: int,
        dropout: float = 0.3,   # ignored
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture 3 — Residual MLP
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """One residual block: output = block(x) + projection(x).

    When in_dim == out_dim the projection is the identity; otherwise a 1×1
    linear maps x to the correct size before adding.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = _mlp_block(in_dim, out_dim, dropout)
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.proj(x)


class ResidualMLPProjector(nn.Module):
    """Residual MLP: each hidden block has a skip connection.

    Identical hyperparameters to MLPProjector — only the skip connections
    differ.  Tends to train more stably with deeper architectures.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        blocks: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            blocks.append(_ResidualBlock(in_dim, h, dropout))
            in_dim = h
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Architecture 4 — Bilinear
# ---------------------------------------------------------------------------

class BilinearProjector(nn.Module):
    """Pure bilinear projection: y = x @ W, no bias, no activation.

    Represents the left factor of a bilinear scoring function
    ``score(text, triple) = (text @ W) · triple``.  Differs from
    ``LinearProjector`` in that there is no bias term, making it a
    strict linear map through the origin.
    ``hidden_dims`` and ``dropout`` are accepted but ignored.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],   # ignored
        output_dim: int,
        dropout: float = 0.3,     # ignored
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture 5 — Low-rank
# ---------------------------------------------------------------------------

class LowRankProjector(nn.Module):
    """Factorised projection W = A @ B with rank << min(input_dim, output_dim).

    Forward: y = x @ A @ B  (no bias, no activation).
    rank is taken from ``hidden_dims[0]`` (default 64 if hidden_dims is empty).
    This constrains capacity more tightly than hidden-layer width while keeping
    the mapping linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,   # ignored
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        rank = hidden_dims[0] if hidden_dims else 64
        self.rank = rank
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, output_dim, bias=False)
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture 6 — Cross-attention
# ---------------------------------------------------------------------------

_CA_N_HEADS = 4
_CA_DEFAULT_D_MODEL = 256


class CrossAttentionProjector(nn.Module):
    """Cross-attention projector: learnable (s, p, o) queries attend over text.

    Three learnable query vectors — one each for the subject, predicate, and
    object intent — cross-attend over the text embedding (treated as a single
    key/value token after a linear projection).  The three attended outputs are
    concatenated and projected to ``output_dim``.

    d_model is taken from ``hidden_dims[0]`` (default 256).  It is rounded
    down to the nearest multiple of ``n_heads`` (4) if needed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        raw_d = hidden_dims[0] if hidden_dims else _CA_DEFAULT_D_MODEL
        # ensure divisibility by n_heads
        d_model = max(_CA_N_HEADS, (raw_d // _CA_N_HEADS) * _CA_N_HEADS)
        self.d_model = d_model

        self.text_proj = nn.Linear(input_dim, d_model)
        # 3 learnable query vectors: one per triple component (s, p, o)
        self.queries = nn.Parameter(torch.empty(3, d_model))
        nn.init.xavier_uniform_(self.queries.unsqueeze(0))  # (1, 3, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, _CA_N_HEADS, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(3 * d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # project text to a single key/value token: (B, 1, d_model)
        kv = self.norm(self.text_proj(x)).unsqueeze(1)
        # expand queries to batch: (B, 3, d_model)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        # cross-attention: queries attend over the single text token
        attn_out, _ = self.attn(q, kv, kv)  # (B, 3, d_model)
        # flatten and project to output space
        return self.out_proj(attn_out.flatten(1))  # (B, output_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ARCHITECTURE_TYPES = ["mlp", "linear", "residual", "bilinear", "low_rank", "cross_attention"]

_REGISTRY = {
    "mlp": MLPProjector,
    "linear": LinearProjector,
    "residual": ResidualMLPProjector,
    "bilinear": BilinearProjector,
    "low_rank": LowRankProjector,
    "cross_attention": CrossAttentionProjector,
}


class BidirectionalProjector(nn.Module):
    """Pair of projectors for bidirectional random training.

    The two directions do not share weights because their input and output
    dimensions differ when the KG triple space is not 768-dimensional.
    """

    def __init__(
        self,
        architecture_type: str,
        text_dim: int,
        triple_dim: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.text_to_kg = build_projector(
            architecture_type=architecture_type,
            input_dim=text_dim,
            hidden_dims=hidden_dims,
            output_dim=triple_dim,
            dropout=dropout,
        )
        self.kg_to_text = build_projector(
            architecture_type=architecture_type,
            input_dim=triple_dim,
            hidden_dims=hidden_dims,
            output_dim=text_dim,
            dropout=dropout,
        )

    def project_text_to_kg(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_to_kg(x)

    def project_kg_to_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.kg_to_text(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_projector(
    architecture_type: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float,
) -> nn.Module:
    """Instantiate a projector by name.

    Args:
        architecture_type: one of ``"mlp"``, ``"linear"``, ``"residual"``,
                           ``"bilinear"``, ``"low_rank"``, ``"cross_attention"``.
        input_dim:         text embedding dimension (e.g. 768).
        hidden_dims:       list of hidden layer widths.
                           ``bilinear`` ignores it;
                           ``low_rank`` uses ``hidden_dims[0]`` as rank (default 64);
                           ``cross_attention`` uses ``hidden_dims[0]`` as d_model (default 256).
        output_dim:        KG triple embedding dimension.
        dropout:           dropout rate (ignored by ``bilinear`` and ``low_rank``).

    Returns:
        An ``nn.Module`` with a ``count_parameters()`` method.
    """
    cls = _REGISTRY.get(architecture_type)
    if cls is None:
        raise ValueError(
            f"Unknown architecture_type '{architecture_type}'. "
            f"Must be one of {ARCHITECTURE_TYPES}."
        )
    return cls(input_dim=input_dim, hidden_dims=hidden_dims,
               output_dim=output_dim, dropout=dropout)


def build_alignment_model(
    training_direction: str,
    architecture_type: str,
    text_dim: int,
    triple_dim: int,
    hidden_dims: List[int],
    dropout: float,
) -> nn.Module:
    """Instantiate the model required by a training direction."""
    if training_direction == "text_to_kg":
        return build_projector(
            architecture_type=architecture_type,
            input_dim=text_dim,
            hidden_dims=hidden_dims,
            output_dim=triple_dim,
            dropout=dropout,
        )
    if training_direction == "kg_to_text":
        return build_projector(
            architecture_type=architecture_type,
            input_dim=triple_dim,
            hidden_dims=hidden_dims,
            output_dim=text_dim,
            dropout=dropout,
        )
    if training_direction == "bidirectional_random":
        return BidirectionalProjector(
            architecture_type=architecture_type,
            text_dim=text_dim,
            triple_dim=triple_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    raise ValueError(
        "training_direction must be one of 'text_to_kg', 'kg_to_text', "
        "or 'bidirectional_random'"
    )
