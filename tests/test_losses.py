import torch

from discourse.losses import masked_task_bce


def test_masked_target_ignored():
    logits = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
    targets = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    masks = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = masked_task_bce(logits, targets, masks)
    # only tox active; should be near 0 because logits tox are high and target 1
    assert loss.item() < 0.1
