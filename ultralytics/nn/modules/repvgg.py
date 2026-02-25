# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RepVGG modules for lightweight backbone networks.

RepVGG uses multi-branch training (3x3+1x1+identity) that fuses into a single
3x3 conv at inference, providing strong channel interaction with TensorRT-friendly
architecture.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


__all__ = (
    "RepVGGBlock",
    "RepVGGStage",
    "RepVGGStem",
)


class RepVGGBlock(nn.Module):
    """RepVGG block with multi-branch training and single-path inference.

    Training: 3x3+BN / 1x1+BN / identity+BN three branches summed.
    Inference: fused into a single 3x3 conv + bias.
    """

    def __init__(self, c1: int, c2: int, s: int = 1):
        """Initialize RepVGG block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
        """
        super().__init__()
        self.deploy = False

        # 3x3 branch
        self.rbr_3x3 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, s, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 1x1 branch
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, bias=False),
            nn.BatchNorm2d(c2),
        )

        # Identity branch (only when c1 == c2 and stride == 1)
        self.rbr_identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-branch forward (training mode)."""
        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Single-path forward (inference mode after fuse)."""
        return self.act(self.rbr_reparam(x))

    def fuse(self):
        """Fuse 3x3/1x1/identity branches into a single 3x3 conv + bias."""
        if self.deploy:
            return
        self.deploy = True

        k3, b3 = self._fuse_bn(self.rbr_3x3)
        k1, b1 = self._fuse_bn(self.rbr_1x1)

        # Pad 1x1 kernel to 3x3
        k1 = nn.functional.pad(k1, [1, 1, 1, 1])

        kernel = k3 + k1
        bias = b3 + b1

        if self.rbr_identity is not None:
            ki, bi = self._fuse_bn_identity(self.rbr_identity)
            kernel = kernel + ki
            bias = bias + bi

        self.rbr_reparam = nn.Conv2d(
            kernel.shape[1], kernel.shape[0], 3,
            stride=self.rbr_3x3[0].stride,
            padding=1, bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches
        del self.rbr_3x3
        del self.rbr_1x1
        if hasattr(self, "rbr_identity") and self.rbr_identity is not None:
            del self.rbr_identity

    @staticmethod
    def _fuse_bn(branch: nn.Sequential):
        """Fuse conv + BN into weight and bias tensors."""
        conv, bn = branch[0], branch[1]
        k = conv.weight
        gamma = bn.weight
        beta = bn.bias
        mu = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = (var + eps).sqrt()
        # fused_weight = k * (gamma / std).reshape(-1, 1, 1, 1)
        fused_k = k * (gamma / std).reshape(-1, 1, 1, 1)
        fused_b = beta - mu * gamma / std
        return fused_k, fused_b

    @staticmethod
    def _fuse_bn_identity(bn: nn.BatchNorm2d):
        """Fuse identity + BN into a 3x3 kernel and bias."""
        c = bn.num_features
        # Identity as 3x3: zeros with center=1 per channel
        k = torch.zeros(c, c, 3, 3, device=bn.weight.device, dtype=bn.weight.dtype)
        for i in range(c):
            k[i, i, 1, 1] = 1.0

        gamma = bn.weight
        beta = bn.bias
        mu = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = (var + eps).sqrt()
        fused_k = k * (gamma / std).reshape(-1, 1, 1, 1)
        fused_b = beta - mu * gamma / std
        return fused_k, fused_b


class RepVGGStage(nn.Module):
    """RepVGG stage containing multiple RepVGG blocks.

    First block uses the given stride for downsampling, remaining blocks use stride=1.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, s: int = 1):
        """Initialize RepVGG stage.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            s (int): Stride for first block.
        """
        super().__init__()
        blocks = [RepVGGBlock(c1, c2, s)]
        for _ in range(n - 1):
            blocks.append(RepVGGBlock(c2, c2, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stage blocks."""
        return self.blocks(x)


class RepVGGStem(nn.Module):
    """RepVGG stem: 3x3 conv stride 2 + BN + ReLU."""

    def __init__(self, c1: int = 3, c2: int = 32):
        """Initialize RepVGG stem.

        Args:
            c1 (int): Input channels (default 3 for RGB).
            c2 (int): Output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stem convolution."""
        return self.conv(x)
