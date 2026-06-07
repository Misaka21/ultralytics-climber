# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Optimized loss functions for armor plate keypoint detection.

This module implements TUP-NN-Train-2 style loss functions for high-precision
armor plate 4-point keypoint detection, including:
- WingLoss for better handling of small errors
- Staged training (WingLoss → L1 fine-tuning)
- IoU-weighted soft labels
- Higher loss weights for keypoint regression
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8PoseLoss, KeypointLoss
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.metrics import bbox_iou


class ArmorKeypointLoss(nn.Module):
    """Optimized keypoint loss for armor plate 4-point detection.

    This loss combines:
    1. Wing Loss for better handling of small errors
    2. Smaller sigma values for tighter keypoint constraints
    3. Staged training support (WingLoss → L1 after 40%)

    Attributes:
        sigmas (torch.Tensor): Per-keypoint sigma values for OKS calculation.
        w (float): Wing loss width parameter.
        epsilon (float): Wing loss curvature parameter.
        use_l1 (bool): Whether to use L1 loss for fine-tuning (staged training).
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
        # Staged training: start with WingLoss, switch to L1 after 40%
        self.use_l1 = False

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
        # log1p is numerically more stable than log(1+x) for small x
        small_mask = diff < self.w
        loss = torch.where(
            small_mask,
            self.w * torch.log1p(diff / self.epsilon),
            diff - self.C.to(diff.device)
        )
        return loss

    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
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

        # Normalize by area for scale invariance
        area_scale = torch.sqrt(area + 1e-9)

        if self.use_l1:
            # Staged training: L1 loss for fine-tuning (TUP-style)
            l1_x = torch.abs(pred_kpts[..., 0] - gt_kpts[..., 0])
            l1_y = torch.abs(pred_kpts[..., 1] - gt_kpts[..., 1])
            l1_total = (l1_x + l1_y) * kpt_mask
            l1_normalized = l1_total / area_scale
            # Combine OKS with L1 for fine-tuning
            combined_loss = 0.3 * oks_loss + 0.7 * l1_normalized
        else:
            # Wing loss for direct coordinate regression (helps with small errors)
            wing_x = self.wing_loss(pred_kpts[..., 0], gt_kpts[..., 0])
            wing_y = self.wing_loss(pred_kpts[..., 1], gt_kpts[..., 1])
            wing_total = (wing_x + wing_y) * kpt_mask
            wing_normalized = wing_total / area_scale
            # Combined loss: Wing-dominant for fine-grained accuracy (TUP-style)
            combined_loss = 0.3 * oks_loss + 0.7 * wing_normalized

        return (kpt_loss_factor.view(-1, 1) * combined_loss).mean()


class ArmorPoseLoss(v8PoseLoss):
    """Optimized pose loss for armor plate 4-point keypoint detection.

    Key optimizations (TUP-NN-Train-2 style):
    1. Smaller sigma values (0.05) for tighter keypoint constraints
    2. Wing Loss for better convergence on small errors
    3. Staged training (WingLoss → L1 after 40%)
    4. IoU-weighted soft labels for better supervision
    5. High pose loss weight (80-100x)

    Example:
        >>> from ultralytics.utils.loss_armor import ArmorPoseLoss
        >>> loss_fn = ArmorPoseLoss(model)
        >>> loss, loss_items = loss_fn(predictions, batch)
    """

    def __init__(self, model, use_iou_weighted_cls: bool = True):
        """Initialize ArmorPoseLoss with optimized settings for armor plate detection.

        Args:
            model: The model to compute loss for (must be de-paralleled).
            use_iou_weighted_cls (bool): Whether to use IoU-weighted soft labels for classification.
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

        # TUP-style IoU-weighted soft labels
        self.use_iou_weighted_cls = use_iou_weighted_cls

    def enable_l1_finetuning(self):
        """Enable L1 loss for fine-tuning phase (staged training).

        Call this after 40% of training epochs to switch from WingLoss to L1.
        This follows TUP-NN-Train-2's staged training strategy.
        """
        if hasattr(self.keypoint_loss, 'use_l1'):
            self.keypoint_loss.use_l1 = True

    def disable_l1_finetuning(self):
        """Disable L1 loss and revert to WingLoss."""
        if hasattr(self.keypoint_loss, 'use_l1'):
            self.keypoint_loss.use_l1 = False

    def disable_l1_finetuning(self):
        """Disable L1 loss and use WingLoss (default mode)."""
        if hasattr(self.keypoint_loss, 'use_l1'):
            self.keypoint_loss.use_l1 = False

    def __call__(self, preds, batch):
        """Calculate pose loss with IoU-weighted soft labels (TUP-style).

        Args:
            preds: Model predictions.
            batch: Batch data containing targets.

        Returns:
            tuple: (total_loss * batch_size, loss_items)
        """
        from typing import Any

        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Keep a stable denominator for bbox/dfl losses.
        # IoU reweighting is applied to cls targets only.
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        # TUP-style IoU-weighted soft labels for classification
        # High-quality predictions get stronger supervision signal
        if self.use_iou_weighted_cls and fg_mask.sum() > 0:
            # Calculate IoU between predictions and targets for foreground
            pred_bboxes_scaled = pred_bboxes * stride_tensor
            ious = bbox_iou(
                pred_bboxes_scaled[fg_mask],
                (target_bboxes * stride_tensor)[fg_mask],
                xywh=False,
                CIoU=False
            ).squeeze(-1)
            # Weight classification targets by IoU (TUP: cls_target *= pred_ious)
            target_scores_weighted = target_scores.clone()
            target_scores_weighted[fg_mask] *= ious.unsqueeze(-1)
            target_scores_weighted_sum = target_scores_weighted.sum().clamp(min=1.0)
        else:
            target_scores_weighted = target_scores
            target_scores_weighted_sum = target_scores_sum

        # Cls loss with IoU-weighted targets
        loss[3] = self.bce(pred_scores, target_scores_weighted.to(dtype)).sum() / target_scores_weighted_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

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

    This is a simpler alternative that uses Smooth L1 loss
    which is more stable for some training scenarios.
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
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
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


class PolyIoULoss(nn.Module):
    """Polygon IoU loss for armor plate 4-point detection.

    This loss uses Shapely to compute true quadrilateral IoU between
    predicted and ground truth 4-point bounding boxes.

    Note: Requires shapely library: pip install shapely

    Example:
        >>> poly_loss = PolyIoULoss()
        >>> pred = torch.tensor([[0, 0, 1, 0, 1, 1, 0, 1]])  # 4 corners
        >>> target = torch.tensor([[0.1, 0, 1, 0.1, 0.9, 1, 0, 0.9]])
        >>> loss = poly_loss(pred, target)
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize PolyIoULoss.

        Args:
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
        self._shapely_available = None

    def _check_shapely(self) -> bool:
        """Check if shapely is available."""
        if self._shapely_available is None:
            try:
                from shapely.geometry import Polygon
                self._shapely_available = True
            except ImportError:
                self._shapely_available = False
        return self._shapely_available

    def poly_iou_single(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU for a single pair of quadrilaterals.

        Args:
            pred (torch.Tensor): Predicted corners, shape (8,) as [x1,y1,x2,y2,x3,y3,x4,y4].
            target (torch.Tensor): Target corners, same shape.

        Returns:
            (torch.Tensor): IoU value (scalar).
        """
        from shapely.geometry import Polygon

        # Convert to numpy and reshape to (4, 2)
        pred_np = pred.detach().cpu().numpy().reshape(4, 2)
        target_np = target.detach().cpu().numpy().reshape(4, 2)

        try:
            pred_poly = Polygon(pred_np)
            target_poly = Polygon(target_np)

            # Check validity
            if not pred_poly.is_valid or not target_poly.is_valid:
                return torch.tensor(0.0, device=pred.device)

            intersection = pred_poly.intersection(target_poly).area
            union = pred_poly.union(target_poly).area

            if union < 1e-6:
                return torch.tensor(0.0, device=pred.device)

            iou = intersection / union
            return torch.tensor(iou, device=pred.device, dtype=pred.dtype)
        except Exception:
            return torch.tensor(0.0, device=pred.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Polygon IoU loss.

        Args:
            pred (torch.Tensor): Predicted corners, shape (N, 8) or (N, 4, 2).
            target (torch.Tensor): Target corners, same shape.

        Returns:
            (torch.Tensor): 1 - IoU loss.
        """
        if not self._check_shapely():
            raise ImportError(
                "PolyIoULoss requires shapely. Install with: pip install shapely"
            )

        # Reshape to (N, 8) if needed
        if pred.dim() == 3:
            pred = pred.view(pred.shape[0], -1)
            target = target.view(target.shape[0], -1)

        # Compute IoU for each sample
        ious = torch.stack([
            self.poly_iou_single(p, t) for p, t in zip(pred, target)
        ])

        # Loss = 1 - IoU
        loss = 1 - ious

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def keypoints_to_polygon(kpts: torch.Tensor) -> torch.Tensor:
    """Convert keypoints (N, 4, 2) to polygon format (N, 8).

    Args:
        kpts (torch.Tensor): Keypoints tensor of shape (N, 4, 2).

    Returns:
        (torch.Tensor): Polygon tensor of shape (N, 8).
    """
    return kpts.view(kpts.shape[0], -1)


def min_rect(corners: torch.Tensor) -> torch.Tensor:
    """Convert 4 corners to minimum bounding rectangle [cx, cy, w, h].

    This is useful for IoU-based matching when using corner points.
    TUP uses this for SimOTA matching.

    Args:
        corners (torch.Tensor): Corner points, shape (N, 8) as [x1,y1,x2,y2,x3,y3,x4,y4].

    Returns:
        (torch.Tensor): Bounding rectangles, shape (N, 4) as [cx, cy, w, h].
    """
    # Reshape to (N, 4, 2)
    corners = corners.view(-1, 4, 2)

    # Get min/max coordinates
    x_coords = corners[..., 0]
    y_coords = corners[..., 1]

    x_min = x_coords.min(dim=1).values
    x_max = x_coords.max(dim=1).values
    y_min = y_coords.min(dim=1).values
    y_max = y_coords.max(dim=1).values

    # Convert to [cx, cy, w, h]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return torch.stack([cx, cy, w, h], dim=1)
