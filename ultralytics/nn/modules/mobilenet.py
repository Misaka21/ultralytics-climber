# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""MobileNetV3 modules for lightweight backbone networks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, autopad

__all__ = (
    "MobileNetV3Block",
    "MobileNetV3Stem",
    "MobileNetV3Stage",
    "SEBlock",
    "HSwish",
    "HSigmoid",
)


class HSigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace: bool = True):
        """Initialize HSigmoid module."""
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard sigmoid activation."""
        return self.relu6(x + 3) / 6


class HSwish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace: bool = True):
        """Initialize HSwish module."""
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard swish activation."""
        return x * self.hsigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    This module implements channel-wise attention mechanism that adaptively
    recalibrates channel features by modeling channel interdependencies.
    """

    def __init__(self, c1: int, reduction: int = 4):
        """Initialize SE block.

        Args:
            c1 (int): Number of input channels.
            reduction (int): Reduction ratio for bottleneck.
        """
        super().__init__()
        c_ = max(c1 // reduction, 8)  # hidden channels (minimum 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c_),
            nn.ReLU(inplace=True),
            nn.Linear(c_, c1),
            HSigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention to input tensor."""
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MobileNetV3Block(nn.Module):
    """MobileNetV3 inverted residual block with optional SE attention.

    This block implements the core building block of MobileNetV3 architecture,
    featuring expansion, depthwise convolution, SE attention, and projection.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        e: float = 1.0,
        se: bool = False,
        act: str = "RE",
    ):
        """Initialize MobileNetV3 block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size for depthwise conv.
            s (int): Stride for depthwise conv.
            e (float): Expansion ratio.
            se (bool): Whether to use SE attention.
            act (str): Activation type ('RE' for ReLU, 'HS' for HSwish).
        """
        super().__init__()
        self.use_res_connect = s == 1 and c1 == c2

        # Determine activation
        activation = HSwish() if act == "HS" else nn.ReLU(inplace=True)

        # Hidden channels
        c_ = int(c1 * e)

        layers = []

        # Expansion phase (only if expansion ratio > 1)
        if e != 1.0:
            layers.extend([
                nn.Conv2d(c1, c_, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_),
                activation,
            ])

        # Depthwise phase
        layers.extend([
            nn.Conv2d(c_, c_, k, s, autopad(k), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            activation,
        ])

        # SE attention
        if se:
            layers.append(SEBlock(c_))

        # Projection phase (linear - no activation)
        layers.extend([
            nn.Conv2d(c_, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MobileNetV3 block transformation."""
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3Stem(nn.Module):
    """MobileNetV3 stem layer (first convolution).

    Initial convolution layer with stride 2 to reduce spatial dimensions.
    """

    def __init__(self, c1: int = 3, c2: int = 16):
        """Initialize MobileNetV3 stem.

        Args:
            c1 (int): Input channels (default 3 for RGB).
            c2 (int): Output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            HSwish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stem convolution."""
        return self.conv(x)


class MobileNetV3Stage(nn.Module):
    """MobileNetV3 stage containing multiple inverted residual blocks.

    This module stacks multiple MobileNetV3 blocks to form a stage,
    optionally with downsampling in the first block.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        k: int = 3,
        s: int = 1,
        e: float = 1.0,
        se: bool = False,
        act: str = "RE",
    ):
        """Initialize MobileNetV3 stage.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks in stage.
            k (int): Kernel size.
            s (int): Stride for first block (for downsampling).
            e (float): Expansion ratio.
            se (bool): Whether to use SE attention.
            act (str): Activation type.
        """
        super().__init__()
        blocks = []

        # First block (may have stride > 1 for downsampling)
        blocks.append(MobileNetV3Block(c1, c2, k, s, e, se, act))

        # Remaining blocks (stride=1, input=output channels)
        for _ in range(n - 1):
            blocks.append(MobileNetV3Block(c2, c2, k, 1, e, se, act))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stage blocks to input."""
        return self.blocks(x)


class MobileNetV3Downsample(nn.Module):
    """Downsampling module using depthwise separable convolution."""

    def __init__(self, c1: int, c2: int):
        """Initialize downsampling module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(c1, c1, 3, 2, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            HSwish(),
            # Pointwise
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling convolution."""
        return self.conv(x)


class MobileNetV3Large(nn.Module):
    """Complete MobileNetV3-Large backbone for feature extraction.

    This module implements the full MobileNetV3-Large architecture
    and returns multi-scale features for detection tasks.
    """

    def __init__(self, c1: int = 3, width_mult: float = 1.0):
        """Initialize MobileNetV3-Large backbone.

        Args:
            c1 (int): Input channels.
            width_mult (float): Width multiplier for channel scaling.
        """
        super().__init__()

        def _make_divisible(v, divisor=8):
            """Make value divisible by divisor."""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def c(channels):
            """Apply width multiplier."""
            return _make_divisible(channels * width_mult)

        # Stem
        self.stem = MobileNetV3Stem(c1, c(16))

        # Stage 1: 1/2 -> 1/4
        self.stage1 = nn.Sequential(
            MobileNetV3Block(c(16), c(16), 3, 1, 1, False, "RE"),
            MobileNetV3Block(c(16), c(24), 3, 2, 4, False, "RE"),  # downsample
            MobileNetV3Block(c(24), c(24), 3, 1, 3, False, "RE"),
        )

        # Stage 2: 1/4 -> 1/8 (P3)
        self.stage2 = nn.Sequential(
            MobileNetV3Block(c(24), c(40), 5, 2, 3, True, "RE"),  # downsample
            MobileNetV3Block(c(40), c(40), 5, 1, 3, True, "RE"),
            MobileNetV3Block(c(40), c(40), 5, 1, 3, True, "RE"),
        )

        # Stage 3: 1/8 -> 1/16 (P4)
        self.stage3 = nn.Sequential(
            MobileNetV3Block(c(40), c(80), 3, 2, 6, False, "HS"),  # downsample
            MobileNetV3Block(c(80), c(80), 3, 1, 2.5, False, "HS"),
            MobileNetV3Block(c(80), c(80), 3, 1, 2.3, False, "HS"),
            MobileNetV3Block(c(80), c(80), 3, 1, 2.3, False, "HS"),
            MobileNetV3Block(c(80), c(112), 3, 1, 6, True, "HS"),
            MobileNetV3Block(c(112), c(112), 3, 1, 6, True, "HS"),
        )

        # Stage 4: 1/16 -> 1/32 (P5)
        self.stage4 = nn.Sequential(
            MobileNetV3Block(c(112), c(160), 5, 2, 6, True, "HS"),  # downsample
            MobileNetV3Block(c(160), c(160), 5, 1, 6, True, "HS"),
            MobileNetV3Block(c(160), c(160), 5, 1, 6, True, "HS"),
        )

        # Store output channels for FPN
        self.out_channels = [c(40), c(112), c(160)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of feature maps at P3 (1/8), P4 (1/16), P5 (1/32) scales.
        """
        x = self.stem(x)
        x = self.stage1(x)

        p3 = self.stage2(x)   # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32

        return [p3, p4, p5]


class MobileNetV3Small(nn.Module):
    """Complete MobileNetV3-Small backbone for feature extraction.

    Smaller and faster variant of MobileNetV3 for edge deployment.
    """

    def __init__(self, c1: int = 3, width_mult: float = 1.0):
        """Initialize MobileNetV3-Small backbone.

        Args:
            c1 (int): Input channels.
            width_mult (float): Width multiplier for channel scaling.
        """
        super().__init__()

        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def c(channels):
            return _make_divisible(channels * width_mult)

        # Stem
        self.stem = MobileNetV3Stem(c1, c(16))

        # Stage 1: 1/2 -> 1/4
        self.stage1 = nn.Sequential(
            MobileNetV3Block(c(16), c(16), 3, 2, 1, True, "RE"),  # downsample
        )

        # Stage 2: 1/4 -> 1/8 (P3)
        self.stage2 = nn.Sequential(
            MobileNetV3Block(c(16), c(24), 3, 2, 4.5, False, "RE"),  # downsample
            MobileNetV3Block(c(24), c(24), 3, 1, 3.67, False, "RE"),
        )

        # Stage 3: 1/8 -> 1/16 (P4)
        self.stage3 = nn.Sequential(
            MobileNetV3Block(c(24), c(40), 5, 2, 4, True, "HS"),  # downsample
            MobileNetV3Block(c(40), c(40), 5, 1, 6, True, "HS"),
            MobileNetV3Block(c(40), c(40), 5, 1, 6, True, "HS"),
            MobileNetV3Block(c(40), c(48), 5, 1, 3, True, "HS"),
            MobileNetV3Block(c(48), c(48), 5, 1, 3, True, "HS"),
        )

        # Stage 4: 1/16 -> 1/32 (P5)
        self.stage4 = nn.Sequential(
            MobileNetV3Block(c(48), c(96), 5, 2, 6, True, "HS"),  # downsample
            MobileNetV3Block(c(96), c(96), 5, 1, 6, True, "HS"),
            MobileNetV3Block(c(96), c(96), 5, 1, 6, True, "HS"),
        )

        self.out_channels = [c(24), c(48), c(96)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features."""
        x = self.stem(x)
        x = self.stage1(x)

        p3 = self.stage2(x)   # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32

        return [p3, p4, p5]
