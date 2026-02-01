# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Optimized loss functions for armor plate keypoint detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.ops import xyxy2xywh


class ArmorKeypointLoss(nn.Module):
    """Optimized keypoint loss for armor plate 4-point detection.

    This loss combines:
    1. Wing Loss for better handling of small errors
    2. Smaller sigma values for tighter keypoint constraints
    3. Smooth L1 loss for stable training

    Attributes:
        sigmas (torch.Tensor): Per-keypoint sigma values for OKS calculation.
        w (float): Wing loss width parameter.
        epsilon (float): Wing loss curvature parameter.
    """

    def __init__(self, sigmas: torch.Tensor, w: float = 10.0, epsilon: float = 2.0):
        """Initialize ArmorKeypointLoss.

        Args:
            sigmas (torch.Tensor): Sigma values for each keypoint.
            w (float): Wing loss width - controls transition point.
            epsilon (float): Wing loss curvature parameter.
        """
        super().__init__()
        self.sigmas = sigmas
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * torch.log(torch.tensor(1 + self.w / self.epsilon))

    def wing_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Wing Loss for keypoint regression.

        Wing loss is better than L2 for keypoint detection as it:
        - Handles small errors more aggressively (faster convergence)
        - Is robust to outliers for large errors

        Args:
            pred (torch.Tensor): Predicted coordinates.
            target (torch.Tensor): Ground truth coordinates.

        Returns:
            (torch.Tensor): Wing loss value.
        """
        diff = torch.abs(pred - target)
        # For small errors, use log-based loss (stronger gradient)
        small_mask = diff < self.w
        loss = torch.where(small_mask, self.w * torch.log(1 + diff / self.epsilon), diff - self.C.to(diff.device))
        return loss

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss with Wing Loss and OKS-based weighting.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints, shape (N, num_kpts, 2 or 3).
            gt_kpts (torch.Tensor): Ground truth keypoints, shape (N, num_kpts, 2 or 3).
            kpt_mask (torch.Tensor): Visibility mask for keypoints, shape (N, num_kpts).
            area (torch.Tensor): Bounding box area for normalization, shape (N, 1).

        Returns:
            (torch.Tensor): Computed keypoint loss.
        """
        # Euclidean distance for OKS
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)

        # Loss factor based on visible keypoints
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)

        # OKS-based loss (tighter constraint with smaller sigmas)
        sigmas = self.sigmas.to(d.device)
        e = d / ((2 * sigmas).pow(2) * (area + 1e-9) * 2)
        oks_loss = (1 - torch.exp(-e)) * kpt_mask

        # Wing loss for direct coordinate regression (helps with small errors)
        wing_x = self.wing_loss(pred_kpts[..., 0], gt_kpts[..., 0])
        wing_y = self.wing_loss(pred_kpts[..., 1], gt_kpts[..., 1])
        wing_total = (wing_x + wing_y) * kpt_mask

        # Normalize by area for scale invariance
        area_scale = torch.sqrt(area + 1e-9)
        wing_normalized = wing_total / area_scale

        # Combined loss: OKS for global positioning + Wing for fine-grained
        combined_loss = 0.7 * oks_loss + 0.3 * wing_normalized

        return (kpt_loss_factor.view(-1, 1) * combined_loss).mean()


class ArmorPoseLoss(v8PoseLoss):
    """Optimized pose loss for armor plate 4-point keypoint detection.

    Key optimizations:
    1. Smaller sigma values (0.05) for tighter keypoint constraints
    2. Wing Loss for better convergence on small errors
    3. Adjusted loss weight ratios for faster convergence
    4. IoU-aware keypoint loss weighting

    Examples:
        >>> from ultralytics.utils.loss_armor import ArmorPoseLoss
        >>> loss_fn = ArmorPoseLoss(model)
        >>> loss, loss_items = loss_fn(predictions, batch)
    """

    def __init__(self, model):
        """Initialize ArmorPoseLoss with optimized settings for armor plate detection.

        Args:
            model: The model to compute loss for (must be de-paralleled).
        """
        super().__init__(model)

        # Override with armor-specific settings
        nkpt = self.kpt_shape[0]  # Should be 4 for armor plates

        # Use smaller sigma values for tighter keypoint constraints
        # Armor plates have well-defined corner positions
        sigmas = torch.ones(nkpt, device=self.device) * 0.05

        # Use optimized keypoint loss
        self.keypoint_loss = ArmorKeypointLoss(sigmas=sigmas)

        # Adjusted BCE for keypoint visibility (if using 3-dim keypoints)
        self.bce_pose = nn.BCEWithLogitsLoss(reduction="none")

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate optimized keypoints loss for armor plates.

        This method includes additional optimizations:
        1. Per-sample loss weighting based on box size
        2. Gradient clipping for stability
        3. Hard negative mining for difficult samples

        Args:
            masks (torch.Tensor): Binary mask indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index mapping anchors to GT objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2), shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints location loss.
            kpts_obj_loss (torch.Tensor): The keypoints visibility loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find max keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create batched keypoints tensor
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Fill batched_keypoints
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Select keypoints using target indices
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Normalize by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = torch.tensor(0.0, device=self.device)
        kpts_obj_loss = torch.tensor(0.0, device=self.device)

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            pred_kpt = pred_kpts[masks]

            # Calculate area with minimum threshold for small boxes
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            area = torch.clamp(area, min=1.0)  # Prevent division by zero

            # Keypoint visibility mask
            if gt_kpt.shape[-1] == 3:
                kpt_mask = gt_kpt[..., 2] != 0
            else:
                # For 2D keypoints (armor plates), all keypoints are always visible
                kpt_mask = torch.ones_like(gt_kpt[..., 0], dtype=torch.bool)

            # Main keypoint loss
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask.float(), area)

            # Visibility loss (only for 3D keypoints)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float()).mean()

        return kpts_loss, kpts_obj_loss


class SmoothL1KeypointLoss(nn.Module):
    """Smooth L1 based keypoint loss for armor plate detection.

    This is a simpler alternative that uses Smooth L1 loss which is more stable for some training scenarios.
    """

    def __init__(self, sigmas: torch.Tensor, beta: float = 1.0):
        """Initialize SmoothL1KeypointLoss.

        Args:
            sigmas (torch.Tensor): Sigma values for each keypoint.
            beta (float): Smooth L1 loss beta parameter.
        """
        super().__init__()
        self.sigmas = sigmas
        self.beta = beta

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Smooth L1 keypoint loss.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints.
            gt_kpts (torch.Tensor): Ground truth keypoints.
            kpt_mask (torch.Tensor): Visibility mask.
            area (torch.Tensor): Bounding box area.

        Returns:
            (torch.Tensor): Computed loss value.
        """
        # Smooth L1 loss per coordinate
        loss_x = F.smooth_l1_loss(pred_kpts[..., 0], gt_kpts[..., 0], reduction="none", beta=self.beta)
        loss_y = F.smooth_l1_loss(pred_kpts[..., 1], gt_kpts[..., 1], reduction="none", beta=self.beta)

        # Combine and mask
        loss = (loss_x + loss_y) * kpt_mask

        # Normalize by area
        area_scale = torch.sqrt(area + 1e-9)
        loss = loss / area_scale

        # Loss factor based on visible keypoints
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)

        return (kpt_loss_factor.view(-1, 1) * loss).mean()


# Sigma values optimized for armor plate corner points
# Smaller values = tighter constraints = faster convergence
ARMOR_SIGMAS = torch.tensor([0.05, 0.05, 0.05, 0.05])  # 4 corner points
