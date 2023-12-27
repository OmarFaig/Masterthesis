from pytorch3d.ops import knn_points, sample_farthest_points
import torch

def index_points(points, idx):
    """
    NOTE: Similar to `gather` and `knn_gather` in this module, but with
    - different order of dimensions
    - arbitrary number of indexed dimensions
    - apparently much faster (cuda, backward pass, determinsic algorithms)

    Parameters
    -----------
    points: tensor with shape (B, N, C)
    idx: tensor (long) with shape (B, S_1, ... S_d)

    Returns
    -------
    points_indexed: tensor (long) with shape (B, S_1, ..., S_d, C)
    """
    B = points.shape[0]
    d = len(idx.shape) - 1
    view_shape = [B] + [1] * d
    batch_idx = torch.arange(B, dtype=torch.long, device=idx.device).view(view_shape).expand_as(idx)
    return points[batch_idx, idx, :]

def grouping_operation(features, idx):
    """
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    return index_points(features.transpose(-2, -1), idx).movedim(-1, -3)
def absolute_or_relative(value, total):
    """Returns the value if integer or multiplies it with total and converts the result to type(total) if float."""
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return type(total)(value * total)
    else:
        raise ValueError(f"Absolute or relative value must be of type int or float, but is: {type(value)}")