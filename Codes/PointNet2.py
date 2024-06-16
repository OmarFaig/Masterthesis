import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points,knn_gather
from utils.pointnet_utils import index_points
import logging
import warnings
logger = logging.getLogger(__name__)

def absolute_or_relative(value, total):
    """
    Returns the value if integer or multiplies it with total and converts the result to type(total) if float.

    Args:
        value (int or float): Value to be converted.
        total (int or float): Total to be used for relative conversion if value is float.

    Returns:
        int or float: The absolute or relative value.
    """
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return type(total)(value * total)
    else:
        raise ValueError(f"Absolute or relative value must be of type int or float, but is: {type(value)}")


def grouping_operation(features, idx):
    """
    Group features using the provided indices.

    Args:
        features (torch.Tensor): (B, C, N) tensor of features to group.
        idx (torch.Tensor): (B, npoint, nsample) tensor containing the indices of features to group with.

    Returns:
        torch.Tensor: (B, C, npoint, nsample) tensor of grouped features.
    """
    return index_points(features.transpose(-2, -1), idx).movedim(-1, -3)


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        """
        Initialize a 2D Convolutional layer.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            kernel_size (tuple): Kernel size of the convolution.
            stride (tuple): Stride of the convolution.
            if_bn (bool): Whether to use batch normalization.
            activation_fn (function): Activation function to apply.
        """
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        """
        Forward pass for the Conv2d layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and activation.
        """
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP_CONV(nn.Module):
    """
    Multi-layer perceptron with 1D convolutional layers.
    """
    def __init__(self, in_channel, layer_dims, bn=None):
        """
        Initialize a multi-layer perceptron with 1D convolutional layers.

        Args:
            in_channel (int): Number of input channels.
            layer_dims (list of int): List of output channels for each layer.
            bn (bool): Whether to use batch normalization.
        """
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
        """
        Forward pass for the MLP_CONV.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the MLP.
        """
        return self.mlp(inputs)



class MLP_Res(nn.Module):
    """
    Residual MLP layer.
    """
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        """
        Initialize a residual MLP layer.

        Args:
            in_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            out_dim (int): Output dimension.
        """
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Forward pass for the MLP_Res layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, out_dim, n).

        Returns:
            torch.Tensor: Output tensor after applying the residual MLP.
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out



class PointNet_SA_Layer(nn.Module):
    """
    PointNet set abstraction layer: sampling + grouping + PointNet layers.
    Based on the paper: PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.
    """
    def __init__(self, in_channel, mlp_channels, *, npoints=None, nsample=None, radius=None, activation=F.relu):
        """
        Initialize the PointNet set abstraction layer.

        Args:
            in_channel (int): Number of input channels.
            mlp_channels (list of int): List of output channels for each MLP layer.
            npoints (int): Number of points to sample.
            nsample (int): Number of points in each local region.
            radius (float): Radius for ball query.
            activation (function): Activation function to apply.
        """
        super(PointNet_SA_Layer, self).__init__()
        self.npoints = npoints
        self.nsample = nsample
        self.radius = radius
        self.activation = activation
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels[:-1]:
            self.convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.norms.append(nn.Identity())
            last_channel = out_channel
        self.last_conv = nn.Conv2d(last_channel, mlp_channels[-1], 1)

    def _sample(self, xyz, npoints):
        """
        Sample points using iterative farthest point sampling (FPS).

        Args:
            xyz (torch.Tensor): Input points [B, C(3), N].
            npoints (int): Number of points to be sampled from the input points.

        Returns:
            torch.Tensor: Sampled points [B, C(3), npoints].
        """
        if npoints is None:
            return None
        N = xyz.shape[-1]
        if N == npoints:
            sampled_xyz = xyz  # sample all the points [B, C, npoints]
        elif N > npoints:
            sampled_xyz = sample_farthest_points(xyz.transpose(-2, -1), K=npoints, random_start_point=True)[0].transpose(-2, -1)
        else:
            raise RuntimeError('NPoints is bigger than the number of input points!')
        return sampled_xyz

    def _group(self, xyz, points, sampled_xyz):
        """
        Group points using ball query or KNN.

        Args:
            xyz (torch.Tensor): Input points (positions) [B, C(3), N].
            points (torch.Tensor): Input points [B, D, N] - features.
            sampled_xyz (torch.Tensor): Points to be grouped (coordinates of the centroids).

        Returns:
            torch.Tensor: Grouped points positions + feature data [B, C+D, S, K].
        """
        if self.nsample is not None:
            if self.radius is not None:
                group_idx = ball_query(p1=sampled_xyz.transpose(-2, -1), p2=xyz.transpose(-2, -1), K=self.nsample, radius=self.radius).idx
                grouped_xyz = index_points(xyz.transpose(-2, -1), group_idx).movedim(-1, -3)
                grouped_xyz -= sampled_xyz.unsqueeze(-1)
                grouped_xyz /= self.radius
            else:
                group_idx = knn_points(sampled_xyz.transpose(-2, -1), xyz.transpose(-2, -1), K=self.nsample, return_sorted=False).idx
                grouped_xyz = index_points(xyz.transpose(-2, -1), group_idx).movedim(-1, -3)
                grouped_xyz -= sampled_xyz.unsqueeze(-1)

            if points is not None:
                grouped_points = index_points(points.transpose(-2, -1), group_idx).movedim(-1, -3)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=1)
            else:
                grouped_points = grouped_xyz
        else:#group all
            grouped_xyz = xyz.unsqueeze(-2)
            if points is not None:
                grouped_points =points.unsqueeze(2)
                grouped_points = torch.cat([grouped_points,grouped_xyz],dim=1)
            else:
                grouped_points = grouped_xyz
        return grouped_points


    #def forward(self):
        #test
    def forward(self, xyz, points):
        """
        Forward pass for the PointNet_SA_Layer.

        Args:
            xyz (torch.Tensor): Input points (positions) [B, C(3), N].
            points (torch.Tensor): Input points [B, D, N] - features.

        Returns:
            torch.Tensor: Output tensor after applying the PointNet_SA_Layer [B, D', npoints].
        """
        if self.npoints is not None:
            xyz = self._sample(xyz, self.npoints)
        grouped_points = self._group(xyz, points, xyz)

        for i, conv in enumerate(self.convs):
            grouped_points = conv(grouped_points)
            grouped_points = self.norms[i](grouped_points)
            grouped_points = self.activation(grouped_points)

        grouped_points = self.last_conv(grouped_points)

        return grouped_points.squeeze(-1)


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float,
                          device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample,
                       device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :,
                                                            pad:nsample + pad]
    return idx.int()

def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz =  sample_farthest_points(xyz.transpose(-2,-1),K=npoint,random_start_point=True)[0].transpose(-2,-1)
  # (B, 3, npoint)gather_operation(xyz
    if idx is None:
       # group_idx = ball_query(p1=sampled_xyz.transpose(-2, -1), p2=xyz.transpose(-2, -1), K=self.nsample,
                              # radius=self.radius).idx

        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self,
                 npoint,
                 nsample,
                 in_channel,
                 mlp,
                 if_bn=True,
                 group_all=False,
                 use_xyz=True,
                 if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(
            Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points



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