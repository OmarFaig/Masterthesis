import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points
from utils.pointnet_utils import index_points
import logging
import warnings
logger = logging.getLogger(__name__)
def absolute_or_relative(value, total):
    """Returns the value if integer or multiplies it with total and converts the result to type(total) if float."""
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return type(total)(value * total)
    else:
        raise ValueError(f"Absolute or relative value must be of type int or float, but is: {type(value)}")


class PointNet_SA_Layer(nn.Module):
    def __init__(self):
        super(PointNet_SA_Layer, self).__init__()
        pass

    def _group(self):
        pass
    def _sample(self):
        pass
    def forward(self):
        #test
        pass