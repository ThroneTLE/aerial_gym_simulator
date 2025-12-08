import torch


def weighted_imitation_loss(policy_actions, teacher_actions, weights):
    """MSE imitation loss with per-sample weights."""
    diff = policy_actions - teacher_actions
    mse = torch.sum(diff * diff, dim=1)
    if weights is not None:
        mse = mse * weights
    return mse.mean()

