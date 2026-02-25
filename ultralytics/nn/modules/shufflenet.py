# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""ShuffleNetV2 modules for lightweight backbone networks.

ShuffleNetV2 uses channel split and channel shuffle operations to enable
efficient cross-channel information exchange with low computational cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = (
    "ShuffleV2Block",
    "ShuffleV2Stage",
    "ShuffleV2Stem",
)


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels across groups: reshape -> transpose -> flatten."""
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleV2Block(nn.Module):
    """ShuffleNetV2 basic block.

    stride=1: channel split -> one branch identity, other branch transform -> concat -> shuffle.
    stride=2: both branches downsample -> concat -> shuffle, output channels = c2.
    """

    def __init__(self, c1: int, c2: int, s: int = 1):
        """Initialize ShuffleV2 block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride (1 or 2).
        """
        super().__init__()
        self.stride = s

        if s == 1:
            # Channel split: half identity, half transform
            branch_c = c2 // 2
            self.branch2 = nn.Sequential(
                # 1x1 conv
                nn.Conv2d(branch_c, branch_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
                # 3x3 depthwise
                nn.Conv2d(branch_c, branch_c, 3, 1, 1, groups=branch_c, bias=False),
                nn.BatchNorm2d(branch_c),
                # 1x1 conv
                nn.Conv2d(branch_c, branch_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
            )
        else:
            # stride=2: both branches downsample
            branch_c = c2 // 2
            # Branch 1: 3x3 dw stride 2 + 1x1
            self.branch1 = nn.Sequential(
                nn.Conv2d(c1, c1, 3, 2, 1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, branch_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
            )
            # Branch 2: 1x1 + 3x3 dw stride 2 + 1x1
            self.branch2 = nn.Sequential(
                nn.Conv2d(c1, branch_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_c, branch_c, 3, 2, 1, groups=branch_c, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.Conv2d(branch_c, branch_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ShuffleV2 block."""
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleV2Stage(nn.Module):
    """ShuffleNetV2 stage containing multiple ShuffleV2 blocks.

    First block uses the given stride for downsampling, remaining blocks use stride=1.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, s: int = 1):
        """Initialize ShuffleV2 stage.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            s (int): Stride for first block.
        """
        super().__init__()
        blocks = [ShuffleV2Block(c1, c2, s)]
        for _ in range(n - 1):
            blocks.append(ShuffleV2Block(c2, c2, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stage blocks."""
        return self.blocks(x)


class ShuffleV2Stem(nn.Module):
    """ShuffleNetV2 stem: 3x3 conv stride 2 + BN + ReLU + MaxPool stride 2 (directly to /4)."""

    def __init__(self, c1: int = 3, c2: int = 24):
        """Initialize ShuffleV2 stem.

        Args:
            c1 (int): Input channels (default 3 for RGB).
            c2 (int): Output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stem convolution and pooling."""
        return self.conv(x)
