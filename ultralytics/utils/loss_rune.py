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

from ultralytics.utils.loss import v8PoseLoss


class CyclicRotationKeypointLoss(nn.Module):
    """Keypoint loss with 4-fold cyclic rotation invariance for rune detection.

    For 8 keypoints arranged in 4 pairs (rune_targeting class):
    - Computes loss for all 4 cyclic rotations (0, 90, 180, 270 degrees)
    - Takes minimum loss as the final result
    - Maintains intra-group order (left point stays left of right point)
    - Maintains inter-group order (counterclockwise sequence preserved)

    4 cyclic rotation index permutations:
    - rot0 (0 deg):   [0,1,2,3,4,5,6,7]  # original
    - rot1 (90 deg):  [2,3,4,5,6,7,0,1]  # shift by 2 positions
    - rot2 (180 deg): [4,5,6,7,0,1,2,3]  # shift by 4 positions
    - rot3 (270 deg): [6,7,0,1,2,3,4,5]  # shift by 6 positions
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

        # Initialize per-sample loss tensor
        all_losses = torch.zeros(n_samples, pred_kpts.shape[1], device=pred_kpts.device)

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
                loss = self._compute_oks_loss_per_sample(pred_full, gt_rotated, mask_rotated, area_full)
                rotation_losses.append(loss)

            # Stack losses: (4, N_full, 8)
            stacked_losses = torch.stack(rotation_losses, dim=0)

            # Sum across keypoints to get total loss per rotation per sample: (4, N_full)
            loss_per_rotation = stacked_losses.sum(dim=2)

            # Find the best rotation for each sample (minimum total loss)
            best_rot_idx = loss_per_rotation.argmin(dim=0)  # (N_full,)

            # Gather the per-keypoint losses from the best rotation for each sample
            n_full = pred_full.shape[0]
            batch_idx = torch.arange(n_full, device=pred_full.device)
            best_losses = stacked_losses[best_rot_idx, batch_idx, :]  # (N_full, 8)

            all_losses[full_visible_mask] = best_losses

        # Process samples with partial visibility (standard loss, no rotation invariance)
        partial_mask = ~full_visible_mask
        if partial_mask.any():
            pred_partial = pred_kpts[partial_mask]
            gt_partial = gt_kpts[partial_mask]
            mask_partial = kpt_mask[partial_mask]
            area_partial = area[partial_mask]

            partial_loss = self._compute_oks_loss_per_sample(pred_partial, gt_partial, mask_partial, area_partial)
            all_losses[partial_mask] = partial_loss

        # Apply kpt_loss_factor and kpt_mask, then take mean (matching original KeypointLoss)
        kpt_loss_factor = kpt_mask.shape[1] / (kpt_mask.sum(dim=1, keepdim=True) + 1e-9)
        return (kpt_loss_factor * all_losses * kpt_mask).mean()

    def _compute_oks_loss_per_sample(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OKS-based keypoint loss per sample per keypoint.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints, shape (N, 8, 2 or 3).
            gt_kpts (torch.Tensor): Ground truth keypoints, shape (N, 8, 2 or 3).
            kpt_mask (torch.Tensor): Visibility mask, shape (N, 8).
            area (torch.Tensor): Bounding box area, shape (N, 1).

        Returns:
            (torch.Tensor): Per-sample per-keypoint loss, shape (N, 8).
        """
        # Euclidean distance squared
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)

        # OKS-based loss
        sigmas = self.sigmas.to(d.device)
        e = d / ((2 * sigmas).pow(2) * (area + 1e-9) * 2)
        oks_loss = 1 - torch.exp(-e)

        return oks_loss


class RunePoseLoss(v8PoseLoss):
    """Pose loss for rune detection with rotation invariance.

    This loss extends v8PoseLoss to use CyclicRotationKeypointLoss for handling
    the 4-fold symmetry of rune targets. Only the keypoint_loss is replaced;
    all other processing remains unchanged from the parent class.
    """

    def __init__(self, model):
        """Initialize RunePoseLoss with rotation-invariant keypoint loss.

        Args:
            model: The model to compute loss for (must be de-paralleled).
        """
        super().__init__(model)

        # Override keypoint_loss with rotation-invariant version
        # Use the same sigmas as the original v8PoseLoss (1/nkpt for non-COCO)
        nkpt = self.kpt_shape[0]  # Should be 8 for rune
        sigmas = torch.ones(nkpt, device=self.device) / nkpt  # 1/8 = 0.125, matching original
        self.keypoint_loss = CyclicRotationKeypointLoss(sigmas=sigmas)
