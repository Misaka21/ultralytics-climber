# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""EfficientFormer modules for lightweight backbone networks.

EfficientFormer combines efficient convolution blocks in early stages with
lightweight pooling-based attention in the last stage, capturing global
geometric relationships while remaining TensorRT-friendly (no softmax).
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = (
    "EfficientFormerBlock",
    "EfficientFormerStage",
    "EfficientFormerStem",
    "PoolingAttention",
)


class PoolingAttention(nn.Module):
    """Pooling-based attention module (TRT-friendly, no softmax).

    Uses average pooling to capture spatial context, producing an attention
    map via simple subtraction (pool - input captures local contrast).
    """

    def __init__(self, c1: int, pool_size: int = 3):
        """Initialize pooling attention.

        Args:
            c1 (int): Number of input/output channels.
            pool_size (int): Pooling kernel size.
        """
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling attention."""
        return self.pool(x) - x


class _FFN(nn.Module):
    """Feed-forward network used in EfficientFormer blocks.

    Two 1x1 convs with GELU activation in between (MLP in spatial domain).
    """

    def __init__(self, c: int, expansion: float = 4.0):
        """Initialize FFN.

        Args:
            c (int): Input/output channels.
            expansion (float): Hidden layer expansion ratio.
        """
        super().__init__()
        c_hidden = int(c * expansion)
        self.fc1 = nn.Conv2d(c, c_hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(c_hidden, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN."""
        return self.fc2(self.act(self.fc1(x)))


class EfficientFormerBlock(nn.Module):
    """EfficientFormer meta block.

    Conv mode: 3x3 depthwise conv token mixer + FFN.
    Attn mode: pooling attention token mixer + FFN.
    Both use residual connections and layer norm (implemented as BN for efficiency).
    """

    def __init__(self, c1: int, c2: int, s: int = 1, use_attn: bool = False):
        """Initialize EfficientFormer block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride (used only for the first block in a stage for downsampling).
            use_attn (bool): Whether to use pooling attention instead of conv mixer.
        """
        super().__init__()
        self.downsample = None
        if s > 1 or c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, 1, bias=False),
                nn.BatchNorm2d(c2),
            )

        # Token mixer
        if use_attn:
            self.token_mixer = PoolingAttention(c2)
        else:
            self.token_mixer = nn.Sequential(
                nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
                nn.BatchNorm2d(c2),
            )

        self.norm1 = nn.BatchNorm2d(c2)

        # FFN
        self.ffn = _FFN(c2)
        self.norm2 = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply EfficientFormer block."""
        if self.downsample is not None:
            x = self.downsample(x)
        x = x + self.norm1(self.token_mixer(x))
        x = x + self.norm2(self.ffn(x))
        return x


class EfficientFormerStage(nn.Module):
    """EfficientFormer stage containing multiple EfficientFormer blocks.

    First block uses the given stride for downsampling, remaining blocks use stride=1.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, s: int = 1, use_attn: bool = False):
        """Initialize EfficientFormer stage.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            s (int): Stride for first block.
            use_attn (bool): Whether blocks use pooling attention.
        """
        super().__init__()
        blocks = [EfficientFormerBlock(c1, c2, s, use_attn)]
        for _ in range(n - 1):
            blocks.append(EfficientFormerBlock(c2, c2, 1, use_attn))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stage blocks."""
        return self.blocks(x)


class EfficientFormerStem(nn.Module):
    """EfficientFormer stem: two 3x3 convs stride 2 each (directly to /4)."""

    def __init__(self, c1: int = 3, c2: int = 48):
        """Initialize EfficientFormer stem.

        Args:
            c1 (int): Input channels (default 3 for RGB).
            c2 (int): Output channels.
        """
        super().__init__()
        c_mid = c2 // 2
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c_mid, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stem convolutions."""
        return self.conv(x)
