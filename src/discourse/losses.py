from __future__ import annotations

import torch.nn.functional as F


def masked_task_bce(logits, targets, masks):
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    counts = masks.sum(dim=0)
    per_task = (elementwise * masks).sum(dim=0) / counts.clamp_min(1)
    active = counts > 0
    if not active.any():
        raise ValueError("Batch contains no valid supervision")
    return per_task[active].mean()


def prediction_consistency(logits_a, logits_b):
    return F.mse_loss(logits_a.sigmoid(), logits_b.sigmoid())


def total_loss(logits, targets, masks, augmented_logits=None, consistency_weight=0.0):
    loss = masked_task_bce(logits, targets, masks)
    if augmented_logits is not None and consistency_weight > 0:
        loss = loss + consistency_weight * prediction_consistency(logits, augmented_logits)
    return loss
