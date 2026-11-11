import torch

def compute_irm_loss(logits, targets, env_masks, criterion, irm_weight=1.0):
    total_loss = 0.0
    total_penalty = 0.0
    valid_envs = 0
    for mask in env_masks:
        if mask.any():
            env_logits = logits[mask]
            env_targets = targets[mask]
            scale = logits.new_ones((), requires_grad=True)
            loss = criterion(env_logits * scale, env_targets)
            grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
            penalty = torch.sum(grad ** 2)
            total_loss += loss
            total_penalty += penalty
            valid_envs += 1
    if valid_envs > 0:
        return total_loss / valid_envs + irm_weight * (total_penalty / valid_envs)
    else:
        return criterion(logits, targets)


def compute_mdgr_loss(logits, targets, multiplicity, criterion, mdgr_weight=1.0):
    if multiplicity.shape != targets.shape or torch.any(multiplicity < 0) or torch.any(multiplicity > 4):
        raise ValueError('Expected disturbance counts in [0,4] aligned with targets')
    disturbed = multiplicity > 0
    normal = multiplicity == 0
    total = len(targets)
    if total == 0:
        raise ValueError('Empty training batch')
    loss = logits.sum() * 0.0
    if disturbed.any():
        counts = multiplicity[disturbed]
        masks = [counts == 1, counts == 2, counts >= 3]
        loss = loss + disturbed.sum() / total * compute_irm_loss(logits[disturbed], targets[disturbed], masks, criterion, mdgr_weight)
    if normal.any():
        loss = loss + normal.sum() / total * criterion(logits[normal], targets[normal])
    return loss
