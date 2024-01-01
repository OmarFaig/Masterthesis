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
class PointNet_SA(nn.Module):
    """
    Implementation of PointNet Set Abstraction Layer
    """

    def __init__(self,in_channel,mlp_channels,*,num_point=None,num_point_centroid_K=None,radius_ball=None,activation=F.relu):
        #logger.debug(call_str())
        super().__init__()
        self.radius_ball = radius_ball
        pass


    def _sample(self,xyz,num_points):
        '''
        Sampling Layer using iterative farthest point sampling(FPS)
        :arg:
        xyz: input point cloud [B,C,N]
        num_points:number of points to sample

        :return:
        sam_xyz: sampled point cloud [B,C,S] - centroids of local regions
        '''
        if num_points==None: #  num_points error
            return None
        N = xyz.shape[-1]
        S = absolute_or_relative(num_points,N)

        if N==S:
             sam_xyz = xyz
        elif N>S:
            #sample_farthest_points - https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.sample_farthest_points"
            # xyz.transpose(-2,-1) B,C,N -> B,N,C
            sam_xyz = sample_farthest_points(xyz.transpose(-2,-1),K=S,random_start_point=True)[0].transpose(-2,-1)
        else:
            raise RuntimeError(f"can not sample more points than are in the point cloud : {N}<{S}")

        return sam_xyz

    def _group(self,xyz,num_points_centroid_K,sam_xyz):
        ''''
        Grouping Layer from PoitNet++
        input:
        xyz: input point cloud [B,C,N]
        sam_xyz : sampled point cloud [B,C,S] - centroids of local regions (if sampled)
        num_points_centroid_K : number of points in the neighborhood of each centroid
        :return:
        groupped_xyz : grouped points [B,C,S]
        '''
        #Ball query
        if self.radius_ball is not None:
            grouped_xyz_idx = ball_query(p1 = sam_xyz.transpose(-2,-1),radius = self.radius_ball, p2 = xyz.transpose(-2,-1),K = num_points_centroid_K).idx # indices of the centroid groups [B,S,K]
            grouped_xyz = index_points(xyz.transpose(-2,-1),grouped_xyz_idx).movedim(-1,-3) # B,C,S,K
            grouped_xyz -= sam_xyz.unsqueeze(-1) # subtract the sampled centroids #  B,C,S,1
            grouped_xyz /= self.radius_ball


    def forward(self):
        pass
