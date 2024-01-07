import torch
import torch.nn as nn
from torch import einsum
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


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out




class PointNet_SA_Layer(nn.Module):
    '''
    PointNet set of abstraction layer : sampling + grouping + PointNet layers
    based on paper : [1] PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space - https://arxiv.org/abs/1706.02413

    '''
    def __init__(self,in_channel,mlp_channels,*,npoints=None,nsample=None,radius=None,activation=F.relu):
        super(PointNet_SA_Layer, self).__init__()
#activation relu ?!
        self.npoints = npoints
        self.nsample = nsample
        self.radius = radius
        self.activation = activation
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels[-1]:
            self.convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.norms.append(nn.BatchNorm1d())#nn.Identity()
            last_channel = out_channel
        self.last_conv = nn.Conv2d(last_channel,mlp_channels[-1],1 )


    def _sample(self,xyz,npoints):
        '''
        Given input points {x1, x2, ..., xn}, we use iterative farthest point sampling (FPS)
        to choose a subset of points {xi1 , xi2 , ..., xim}, such that xij is the most distant point (in metric
        distance) from the set {xi1 , xi2 , ..., xij1 } with regard to the rest points.
        Args:
            xyz (tensor): input points [B,C(3),N]
            npoints (int): number of points to be sampled from the input points
        Return:
            sampled_xyz : sampled points [B,C(3),npoints]

        '''
        N = xyz.shape[-1]
        if N==npoints:
            sampled_xyz = xyz # sample all the points [B,C,npoints]
        elif N>npoints:
            sampled_xyz = sample_farthest_points(xyz.transpose(-2,-1),K=npoints,random_start_point=True)[0].transpose(-2,-1)
        else: raise RuntimeError(f'NPoints is bigger than the number of input points !')
        return sampled_xyz
    def _group(self,xyz,points,sampled_xyz):
        '''
        The input to this layer is a point set of size N x (d + C) and the coordinates of
        aset of centroids of size N' x d. The output are groups of point sets of size N' x K x (d + C),
        where each group corresponds to a local region and K is the number of points in the neighborhood of
        ccentroid points. Note that K varies across groups but the succeeding PointNet layer is able to convert
        flexible number of points into a fixed length local region feature vector.
        3 ways of grouping : KNN -  Ball Query -  group all
        Input:
        xyz: inout points(positions) [B,C(3),N] or (Nx(d+C)
        points : inout points [B,D,N]  - features
        sampled_xyz: points to be grouped (coordinates of the centroids)
        :return
        grouped_points : grouped points positions + feature data , [B,C+D,S,K]
        '''
        if self.nsample is not None:#sample and group by ball query or knn
            if self.radius is not None:
                # Ball Query
                group_idx=ball_query(p1=sampled_xyz, p2=xyz, K=self.nsample, radius=self.radius).idx
                grouped_xyz = index_points(xyz.transpose(-2,-1), group_idx).movedim(-1,-3)
                grouped_xyz -= sampled_xyz.unsqueeze(-1)
                grouped_xyz /= self.radius
            else:
                #KNN
                grouped_xyz = knn_points(sampled_xyz.transpose(-2,-1),xyz.transpose(-2,-1), K=self.nsample,return_sorted=False).idx
                grouped_xyz = index_points(xyz.transpose(-2,-1),grouped_xyz).movedim(-1,-3)
                grouped_xyz -=sampled_xyz.unsqueeze(-1)

            if points is not None:
                grouped_points = index_points(points.transpose(-2,-1),group_idx).movedim(-1,-3)
                grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-3)
            else:
                grouped_points = grouped_xyz
        else:#group all
            grouped_xyz = xyz.unsqueeze(-2)
            if points is not None:
                grouped_points =points.unsqueeze(-2)
                grouped_points = torch.cat([grouped_points,grouped_xyz],dim=-3)
            else:
                grouped_points = grouped_xyz
        return grouped_points


    def forward(self):
        #test
        def forward(self, xyz, points=None, sampled_xyz=None, npoint=None):
            """
            Args:
                xyz: Tensor, (B, 3, N)
                points: Tensor, (B, f, N)

            Returns:
                new_xyz: Tensor, (B, 3, npoint)
                new_points: Tensor, (B, mlp[-1], npoint)
            """
            if npoint is None:
                npoint = self.npoint
            assert (npoint is None) == (self.nsample is None)  # both None => group all, otherwise sample and group

            if sampled_xyz is None:
                sampled_xyz = self._sample(xyz, npoint)  # (B, C, S)
            grouped_points = self._group(xyz, points, sampled_xyz)  # (B, C+D, S, K)
            for conv, norm in zip(self.convs, self.norms):
                grouped_points = self.activation(norm(conv(grouped_points)))
            grouped_points = self.last_conv(grouped_points)
            new_points = torch.max(grouped_points, -1)[0]  # (B, D', S)

            return sampled_xyz, new_points



class vTransformer(nn.Module):
    def __init__(self,
                 in_channel,
                 dim=256,
                 n_knn=16,
                 pos_hidden_dim=64,
                 attn_hidden_multiplier=4):
        super(vTransformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1),
                                     nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(),
                                     nn.Conv2d(pos_hidden_dim, dim, 1))

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1))

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1)#.contiguous()
        _, idx_knn, _ = knn_points(pos_flipped, pos_flipped, K=self.n_knn, return_sorted=False)
        key = self.conv_key(x)  # (B, dim, N)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape(
            (b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        # knn value is correct
        value = grouping_operation(value,
                                   idx_knn) + pos_embedding  # (B, dim, N, k)

        agg = einsum('b c i j, b c i j -> b c i', attention,
                     value)  # (B, dim, N)
        y = self.linear_end(agg)  # (B, in_dim, N)

        return y + identity
