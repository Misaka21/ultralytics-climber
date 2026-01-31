# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Rotation-invariant loss functions for rune keypoint detection.

This module implements cyclic rotation matching loss for 4-fold symmetric rune targets:
- CyclicRotationKeypointLoss: Computes min loss across 4 rotations for 8 keypoints
- RunePoseLoss: Pose loss with rotation invariance for rune_targeting class

The rune has 4-fold central symmetry with 8 keypoints arranged in 4 pairs.
Due to this symmetry, the network may confuse the rotational order (0/90/180/270 degrees).
This loss function handles the ambiguity by computing loss for all 4 rotations and taking the minimum.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8PoseLoss, KeypointLoss
from ultralytics.utils.ops import xyxy2xywh


class CyclicRotationKeypointLoss(nn.Module):
    """Keypoint loss with 4-fold cyclic rotation invariance for rune detection.

    For 8 keypoints arranged in 4 pairs (rune_targeting class):
    - Computes loss for all 4 cyclic rotations (0, 90, 180, 270 degrees)
    - Takes minimum loss as the final result
    - Maintains intra-group order (left point stays left of right point)
    - Maintains inter-group order (counterclockwise sequence preserved)

    Keypoint grouping:
    - Group 0: [kpt7, kpt0]  # kpt7 left, kpt0 right
    - Group 1: [kpt1, kpt2]  # kpt1 left, kpt2 right
    - Group 2: [kpt3, kpt4]  # kpt3 left, kpt4 right
    - Group 3: [kpt5, kpt6]  # kpt5 left, kpt6 right

    4 cyclic rotation index permutations:
    - rot0 (0 deg):   [0,1,2,3,4,5,6,7]  # original
    - rot1 (90 deg):  [2,3,4,5,6,7,0,1]  # shift by 2 positions
    - rot2 (180 deg): [4,5,6,7,0,1,2,3]  # shift by 4 positions
    - rot3 (270 deg): [6,7,0,1,2,3,4,5]  # shift by 6 positions

    Attributes:
        sigmas (torch.Tensor): Per-keypoint sigma values for OKS calculation.
        ROTATIONS (list): Index permutations for 4 cyclic rotations.
    """

    ROTATIONS = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # 0 deg - original
        [2, 3, 4, 5, 6, 7, 0, 1],  # 90 deg - shift by 2
        [4, 5, 6, 7, 0, 1, 2, 3],  # 180 deg - shift by 4
        [6, 7, 0, 1, 2, 3, 4, 5],  # 270 deg - shift by 6
    ]

    def __init__(self, sigmas: torch.Tensor):
        """Initialize CyclicRotationKeypointLoss.

        Args:
            sigmas (torch.Tensor): Sigma values for each keypoint (length 8).
        """
        super().__init__()
        self.sigmas = sigmas

    def _compute_oks_loss(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OKS-based keypoint loss for a single rotation.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints, shape (N, 8, 2 or 3).
            gt_kpts (torch.Tensor): Ground truth keypoints, shape (N, 8, 2 or 3).
            kpt_mask (torch.Tensor): Visibility mask, shape (N, 8).
            area (torch.Tensor): Bounding box area, shape (N, 1).

        Returns:
            (torch.Tensor): Per-sample loss, shape (N,).
        """
        # Euclidean distance squared
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)

        # Loss factor based on visible keypoints
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)

        # OKS-based loss
        sigmas = self.sigmas.to(d.device)
        e = d / ((2 * sigmas).pow(2) * (area + 1e-9) * 2)
        oks_loss = (1 - torch.exp(-e)) * kpt_mask

        # Per-sample loss (sum over keypoints)
        per_sample_loss = (kpt_loss_factor.view(-1, 1) * oks_loss).sum(dim=1)

        return per_sample_loss

    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rotation-invariant keypoint loss.

        For samples with all 8 keypoints visible (rune_targeting), computes loss
        for all 4 rotations and takes the minimum. For other samples, uses standard loss.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints, shape (N, 8, 2 or 3).
            gt_kpts (torch.Tensor): Ground truth keypoints, shape (N, 8, 2 or 3).
            kpt_mask (torch.Tensor): Visibility mask, shape (N, 8).
            area (torch.Tensor): Bounding box area, shape (N, 1).

        Returns:
            (torch.Tensor): Scalar loss value.
        """
        n_samples = pred_kpts.shape[0]
        if n_samples == 0:
            return torch.tensor(0.0, device=pred_kpts.device)

        # Check which samples have all 8 keypoints visible (rune_targeting)
        visible_count = kpt_mask.sum(dim=1)
        full_visible_mask = visible_count == 8  # Only apply rotation invariance to these

        # Initialize loss tensor
        final_loss = torch.zeros(n_samples, device=pred_kpts.device)

        # Process samples with all 8 keypoints visible (rotation-invariant)
        if full_visible_mask.any():
            pred_full = pred_kpts[full_visible_mask]
            gt_full = gt_kpts[full_visible_mask]
            mask_full = kpt_mask[full_visible_mask]
            area_full = area[full_visible_mask]

            # Compute loss for all 4 rotations
            rotation_losses = []
            for rot_idx in self.ROTATIONS:
                # Reorder ground truth keypoints according to rotation
                gt_rotated = gt_full[:, rot_idx, :]
                mask_rotated = mask_full[:, rot_idx]
                loss = self._compute_oks_loss(pred_full, gt_rotated, mask_rotated, area_full)
                rotation_losses.append(loss)

            # Stack and take minimum across rotations for each sample
            stacked_losses = torch.stack(rotation_losses, dim=0)  # (4, N_full)
            min_loss, _ = stacked_losses.min(dim=0)  # (N_full,)

            final_loss[full_visible_mask] = min_loss

        # Process samples with partial visibility (standard loss, no rotation invariance)
        partial_mask = ~full_visible_mask
        if partial_mask.any():
            pred_partial = pred_kpts[partial_mask]
            gt_partial = gt_kpts[partial_mask]
            mask_partial = kpt_mask[partial_mask]
            area_partial = area[partial_mask]

            partial_loss = self._compute_oks_loss(pred_partial, gt_partial, mask_partial, area_partial)
            final_loss[partial_mask] = partial_loss

        return final_loss.mean()


class RunePoseLoss(v8PoseLoss):
    """Pose loss for rune detection with rotation invariance.

    This loss extends v8PoseLoss to use CyclicRotationKeypointLoss for handling
    the 4-fold symmetry of rune targets. When all 8 keypoints are visible
    (rune_targeting class), the loss is computed across all 4 rotations and
    the minimum is used as the final loss.

    Example:
        >>> from ultralytics.utils.loss_rune import RunePoseLoss
        >>> loss_fn = RunePoseLoss(model)
        >>> loss, loss_items = loss_fn(predictions, batch)
    """

    def __init__(self, model):
        """Initialize RunePoseLoss with rotation-invariant keypoint loss.

        Args:
            model: The model to compute loss for (must be de-paralleled).
        """
        super().__init__(model)

        # Override with rune-specific settings
        nkpt = self.kpt_shape[0]  # Should be 8 for rune

        # Use smaller sigma values for tighter keypoint constraints
        sigmas = torch.ones(nkpt, device=self.device) * 0.05

        # Use rotation-invariant keypoint loss
        self.keypoint_loss = CyclicRotationKeypointLoss(sigmas=sigmas)

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
        """Calculate keypoints loss with rotation invariance for rune targets.

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
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
            device=keypoints.device
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
            area = torch.clamp(area, min=1.0)

            # Keypoint visibility mask
            if gt_kpt.shape[-1] == 3:
                kpt_mask = gt_kpt[..., 2] != 0
            else:
                # For 2D keypoints, all keypoints are always visible
                kpt_mask = torch.ones_like(gt_kpt[..., 0], dtype=torch.bool)

            # Main keypoint loss (with rotation invariance)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask.float(), area)

            # Visibility loss (only for 3D keypoints)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float()).mean()

        return kpts_loss, kpts_obj_loss
