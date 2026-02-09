# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Rotation-invariant loss functions for rune keypoint detection."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import make_anchors


class CyclicRotationKeypointLoss(nn.Module):
    """Rotation-invariant keypoint loss with staged Wing -> L1 fine-tuning."""

    ROTATIONS = (
        (0, 1, 2, 3, 4, 5, 6, 7),  # 0 deg
        (2, 3, 4, 5, 6, 7, 0, 1),  # 90 deg
        (4, 5, 6, 7, 0, 1, 2, 3),  # 180 deg
        (6, 7, 0, 1, 2, 3, 4, 5),  # 270 deg
    )

    def __init__(self, sigmas: torch.Tensor, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.sigmas = sigmas
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)
        self.use_l1 = False

    def wing_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        return torch.where(
            diff < self.w,
            self.w * torch.log1p(diff / self.epsilon),
            diff - self.C,
        )

    def _compute_component_loss(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        mask = kpt_mask.float()
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        e = d / ((2 * self.sigmas.to(d.device)).pow(2) * (area + 1e-9) * 2)
        oks_loss = (1 - torch.exp(-e)) * mask

        area_scale = torch.sqrt(area + 1e-9)
        if self.use_l1:
            l1_loss = (torch.abs(pred_kpts[..., 0] - gt_kpts[..., 0]) + torch.abs(pred_kpts[..., 1] - gt_kpts[..., 1])) * mask
            reg_loss = l1_loss / area_scale
        else:
            wing = (self.wing_loss(pred_kpts[..., 0], gt_kpts[..., 0]) + self.wing_loss(pred_kpts[..., 1], gt_kpts[..., 1])) * mask
            reg_loss = wing / area_scale

        return 0.3 * oks_loss + 0.7 * reg_loss

    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        n_samples, nkpt = pred_kpts.shape[:2]
        if n_samples == 0:
            return torch.tensor(0.0, device=pred_kpts.device)

        visible_count = (kpt_mask > 0).sum(dim=1)
        full_visible_mask = visible_count == nkpt
        all_losses = torch.zeros(n_samples, nkpt, device=pred_kpts.device)

        if full_visible_mask.any():
            pred_full = pred_kpts[full_visible_mask]
            gt_full = gt_kpts[full_visible_mask]
            mask_full = kpt_mask[full_visible_mask]
            area_full = area[full_visible_mask]

            rotation_losses = []
            for rot_idx in self.ROTATIONS:
                gt_rotated = gt_full[:, rot_idx, :]
                mask_rotated = mask_full[:, rot_idx]
                rotation_losses.append(self._compute_component_loss(pred_full, gt_rotated, mask_rotated, area_full))

            stacked_losses = torch.stack(rotation_losses, dim=0)  # (4, n_full, nkpt)
            best_rot_idx = stacked_losses.sum(dim=2).argmin(dim=0)
            batch_idx = torch.arange(pred_full.shape[0], device=pred_kpts.device)
            all_losses[full_visible_mask] = stacked_losses[best_rot_idx, batch_idx, :]

        partial_mask = ~full_visible_mask
        if partial_mask.any():
            all_losses[partial_mask] = self._compute_component_loss(
                pred_kpts[partial_mask],
                gt_kpts[partial_mask],
                kpt_mask[partial_mask],
                area[partial_mask],
            )

        kpt_loss_factor = nkpt / ((kpt_mask > 0).sum(dim=1, keepdim=True) + 1e-9)
        return (kpt_loss_factor * all_losses).mean()


class RunePoseLoss(v8PoseLoss):
    """Rune pose loss with rotation-invariant keypoint matching and TUP-style tricks."""

    def __init__(self, model, use_iou_weighted_cls: bool = True):
        super().__init__(model)
        nkpt = self.kpt_shape[0]
        sigmas = torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = CyclicRotationKeypointLoss(sigmas=sigmas)
        self.bce_pose = nn.BCEWithLogitsLoss()
        self.use_iou_weighted_cls = use_iou_weighted_cls
        ignore = getattr(model, "pose_ignore_classes", set()) or set()
        self.pose_ignore_classes = {int(c) for c in ignore}

    def enable_l1_finetuning(self):
        if hasattr(self.keypoint_loss, "use_l1"):
            self.keypoint_loss.use_l1 = True

    def disable_l1_finetuning(self):
        if hasattr(self.keypoint_loss, "use_l1"):
            self.keypoint_loss.use_l1 = False

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        target_scores_for_cls = target_scores
        target_scores_for_cls_sum = target_scores_sum

        if self.use_iou_weighted_cls and fg_mask.sum() > 0:
            pred_bboxes_scaled = pred_bboxes * stride_tensor
            ious = bbox_iou(
                pred_bboxes_scaled[fg_mask],
                target_bboxes[fg_mask],
                xywh=False,
                CIoU=False,
            ).squeeze(-1).clamp_(0)
            target_scores_for_cls = target_scores.clone()
            target_scores_for_cls[fg_mask] *= ious.unsqueeze(-1)
            target_scores_for_cls_sum = target_scores_for_cls.sum().clamp(min=1.0)

        loss[3] = self.bce(pred_scores, target_scores_for_cls.to(dtype)).sum() / target_scores_for_cls_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch_idx,
                stride_tensor,
                target_bboxes,
                pred_kpts,
                gt_labels,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl

        return loss * batch_size, loss.detach()

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate keypoint losses while ignoring classes without keypoint supervision."""
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        selected_labels = gt_labels.squeeze(-1).long().gather(1, target_gt_idx)
        pose_masks = masks
        if self.pose_ignore_classes:
            ignore = torch.tensor(sorted(self.pose_ignore_classes), device=selected_labels.device, dtype=selected_labels.dtype)
            pose_masks = masks & (~torch.isin(selected_labels, ignore))

        kpts_loss = torch.tensor(0.0, device=keypoints.device)
        kpts_obj_loss = torch.tensor(0.0, device=keypoints.device)
        if pose_masks.any():
            gt_kpt = selected_keypoints[pose_masks]
            area = xyxy2xywh(target_bboxes[pose_masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[pose_masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss
