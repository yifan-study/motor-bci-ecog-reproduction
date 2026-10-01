"""Loss function registry for training.

Provides named loss functions used across all models.
"""

import torch.nn.functional as F


def mse_loss(pred, target, **kwargs):
    """Plain MSE loss."""
    return F.mse_loss(pred, target)


def mse_cosine_loss(pred, target, mse_weight=0.5):
    """Lomtev combined loss: 0.5*MSE + 0.5*(1 - cosine_similarity).

    Parameters
    ----------
    pred, target : torch.Tensor, shape (batch, channels, time)
    mse_weight : float
    """
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    return mse_weight * mse + (1 - mse_weight) * (1 - cos)


LOSS_REGISTRY = {
    "mse": mse_loss,
    "mse_cosine": mse_cosine_loss,
}


def get_loss_fn(name):
    """Look up a loss function by name.

    Parameters
    ----------
    name : str
        One of the keys in LOSS_REGISTRY.

    Returns
    -------
    callable
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss {name!r}. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]
