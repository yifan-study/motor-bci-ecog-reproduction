"""Evaluation metrics for decoding models.

Generalised to arbitrary output channels (fingers, joystick axes, etc.).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def pearson_r_per_channel(pred, target):
    """Pearson correlation per output channel.

    Parameters
    ----------
    pred, target : np.ndarray, shape (channels, time)

    Returns
    -------
    np.ndarray, shape (channels,) — Pearson r per channel
    """
    rs = []
    for i in range(pred.shape[0]):
        p, t = pred[i], target[i]
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            rs.append(0.0)
        else:
            r = np.corrcoef(p, t)[0, 1]
            rs.append(r if np.isfinite(r) else 0.0)
    return np.array(rs)


def smooth_predictions(pred, sigma):
    """Apply Gaussian smoothing along the time axis.

    Parameters
    ----------
    pred : np.ndarray, shape (channels, time)
    sigma : float
        Gaussian kernel standard deviation. 0 = no-op.

    Returns
    -------
    np.ndarray, same shape as pred
    """
    if sigma <= 0:
        return pred
    out = pred.copy()
    for i in range(out.shape[0]):
        out[i] = gaussian_filter1d(out[i], sigma=sigma)
    return out
