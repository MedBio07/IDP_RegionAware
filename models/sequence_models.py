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


class LowRankResidualAdapter(nn.Module):
    """Low-rank residual adapter for frozen-representation specialization."""

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.down(self.norm(x))
        z = self.up(self.dropout(F.gelu(z)))
        return (x + z) * mask.unsqueeze(-1)


class RegionAdapterMoETCN(nn.Module):
    """Region-specialized low-rank adapters with a learned MoE residue gate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.15,
        kernels: tuple[int, ...] = (3, 7, 15),
        adapter_dim: int = 32,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.gate_temperature = gate_temperature
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
        self.region_adapters = nn.ModuleList(
            [
                LowRankResidualAdapter(
                    hidden_dim=hidden_dim,
                    adapter_dim=adapter_dim,
                    dropout=dropout,
                )
                for _ in range(4)
            ]
        )
        self.generic_head = nn.Linear(hidden_dim, 1)
        self.expert_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(4)])
        self.gate_head = nn.Linear(hidden_dim, 4)
        self.auxiliary_head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_projection(self.input_norm(x))
        h = F.gelu(h) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)
        temperature = max(float(self.gate_temperature), 1.0e-6)
        gate_logits = self.gate_head(h) / temperature
        gate_weights = torch.softmax(gate_logits, dim=-1)
        adapted_states = torch.stack([adapter(h, mask) for adapter in self.region_adapters], dim=2)
        expert_logits = torch.stack(
            [head(adapted_states[:, :, index, :]).squeeze(-1) for index, head in enumerate(self.expert_heads)],
            dim=-1,
        )
        generic_logits = self.generic_head(h).squeeze(-1)
        expert_delta = torch.sum(expert_logits * gate_weights, dim=-1)
        disorder_logits = generic_logits + expert_delta
        auxiliary_logits = self.auxiliary_head(h)
        return {
            "disorder_logits": disorder_logits,
            "generic_logits": generic_logits,
            "expert_logits": expert_logits,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "auxiliary_logits": auxiliary_logits,
        }


class FactorizedRegionAdapterMoETCN(RegionAdapterMoETCN):
    """RegionAdapterMoETCN with factorized length and location gates.

    The four experts retain the historical order ``SDR, LDR, terminal,
    internal``.  The public ``gate_weights`` tensor keeps that order and is
    scaled by 0.5 within each factor so existing four-way region targets and
    losses remain valid.
    """

    length_gate_names = ("sdr", "ldr")
    location_gate_names = ("terminal_idr", "internal_idr")

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.15,
        kernels: tuple[int, ...] = (3, 7, 15),
        adapter_dim: int = 32,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            kernels=kernels,
            adapter_dim=adapter_dim,
            gate_temperature=gate_temperature,
        )
        del self.gate_head
        self.length_gate_head = nn.Linear(hidden_dim, 2)
        self.location_gate_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_projection(self.input_norm(x))
        h = F.gelu(h) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)

        temperature = max(float(self.gate_temperature), 1.0e-6)
        length_gate_logits = self.length_gate_head(h) / temperature
        location_gate_logits = self.location_gate_head(h) / temperature
        length_gate_weights = torch.softmax(length_gate_logits, dim=-1)
        location_gate_weights = torch.softmax(location_gate_logits, dim=-1)
        gate_weights = torch.cat(
            (0.5 * length_gate_weights, 0.5 * location_gate_weights),
            dim=-1,
        )

        adapted_states = torch.stack([adapter(h, mask) for adapter in self.region_adapters], dim=2)
        expert_logits = torch.stack(
            [head(adapted_states[:, :, index, :]).squeeze(-1) for index, head in enumerate(self.expert_heads)],
            dim=-1,
        )
        generic_logits = self.generic_head(h).squeeze(-1)
        length_delta = torch.sum(expert_logits[..., :2] * length_gate_weights, dim=-1)
        location_delta = torch.sum(expert_logits[..., 2:] * location_gate_weights, dim=-1)
        expert_delta = 0.5 * (length_delta + location_delta)
        disorder_logits = generic_logits + expert_delta
        auxiliary_logits = self.auxiliary_head(h)
        return {
            "disorder_logits": disorder_logits,
            "generic_logits": generic_logits,
            "expert_logits": expert_logits,
            "gate_logits": torch.cat((length_gate_logits, location_gate_logits), dim=-1),
            "length_gate_logits": length_gate_logits,
            "location_gate_logits": location_gate_logits,
            "length_gate_weights": length_gate_weights,
            "location_gate_weights": location_gate_weights,
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
