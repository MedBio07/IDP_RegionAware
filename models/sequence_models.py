"""Sequence models for residue-level intrinsic disorder prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiKernelTCNBlock(nn.Module):
    """Residual 1D convolution block with several receptive-field sizes."""

    def __init__(
        self,
        hidden_dim: int,
        kernels: tuple[int, ...] = (3, 7, 15),
        dilation: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel,
                    padding=dilation * (kernel // 2),
                    dilation=dilation,
                )
                for kernel in kernels
            ]
        )
        self.mix = nn.Conv1d(hidden_dim * len(kernels), hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x).transpose(1, 2)
        z = torch.cat([conv(z) for conv in self.convs], dim=1)
        z = self.mix(z).transpose(1, 2)
        z = self.dropout(F.gelu(z))
        return (residual + z) * mask.unsqueeze(-1)


class RegionAwareTCN(nn.Module):
    """Frozen-feature TCN with disorder, expert, gate, and auxiliary region heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.15,
        kernels: tuple[int, ...] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MultiKernelTCNBlock(
                    hidden_dim=hidden_dim,
                    kernels=kernels,
                    dilation=2 ** index,
                    dropout=dropout,
                )
                for index in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.generic_head = nn.Linear(hidden_dim, 1)
        self.expert_head = nn.Linear(hidden_dim, 4)
        self.gate_head = nn.Linear(hidden_dim, 4)
        self.auxiliary_head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_projection(self.input_norm(x))
        h = F.gelu(h) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)
        generic_logits = self.generic_head(h).squeeze(-1)
        expert_logits = self.expert_head(h)
        gate_weights = torch.softmax(self.gate_head(h), dim=-1)
        expert_delta = torch.sum(expert_logits * gate_weights, dim=-1)
        disorder_logits = generic_logits + expert_delta
        auxiliary_logits = self.auxiliary_head(h)
        return {
            "disorder_logits": disorder_logits,
            "generic_logits": generic_logits,
            "expert_logits": expert_logits,
            "gate_weights": gate_weights,
            "auxiliary_logits": auxiliary_logits,
        }


class GenericTCN(nn.Module):
    """Frozen-feature TCN without region experts or auxiliary heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.15,
        kernels: tuple[int, ...] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MultiKernelTCNBlock(
                    hidden_dim=hidden_dim,
                    kernels=kernels,
                    dilation=2 ** index,
                    dropout=dropout,
                )
                for index in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.disorder_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_projection(self.input_norm(x))
        h = F.gelu(h) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)
        return {"disorder_logits": self.disorder_head(h).squeeze(-1)}


class AuxiliaryTCN(nn.Module):
    """Frozen-feature TCN with auxiliary region heads but no expert gate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.15,
        kernels: tuple[int, ...] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MultiKernelTCNBlock(
                    hidden_dim=hidden_dim,
                    kernels=kernels,
                    dilation=2 ** index,
                    dropout=dropout,
                )
                for index in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.disorder_head = nn.Linear(hidden_dim, 1)
        self.auxiliary_head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_projection(self.input_norm(x))
        h = F.gelu(h) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)
        return {
            "disorder_logits": self.disorder_head(h).squeeze(-1),
            "auxiliary_logits": self.auxiliary_head(h),
        }
