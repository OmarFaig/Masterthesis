
import torch
from torch import nn, einsum
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points
#from pointnet_utils import furthest_point_sample, \
#    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
#from pointnet_utils import gather_operation, grouping_operation,index_points,absolute_or_relative
from pytorch3d.ops import knn_points, sample_farthest_points

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
def gather_operation(points, index):
    """Gathers the indexed K points.

    Parameters
    ----------
    points: tensor with shape (B, C, N)
    index: tensor (long) with shape (B, K)

    Returns
    -------
    points_indexed : tensor with shape (B, C, K)
    """
    index_expanded = index.unsqueeze(-2).expand(-1, points.shape[-2], -1)  # (B, C, K)
    points_indexed = torch.gather(points, -1, index_expanded)
    return points_indexed


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

def query_ball_point(radius, nsample, xyz, new_xyz, return_pts_count=False):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    group_idx = ball_query(p1=new_xyz, p2=xyz, K=nsample, radius=radius).idx  # (B, S, nsample)
    group_first = group_idx[..., 0:1].expand_as(group_idx)  # (B, S, nsample)
    mask = group_idx == -1

    group_idx[mask] = group_first[mask]

    if return_pts_count:
        pts_per_ball = torch.sum(~mask, dim=-1)
        return group_idx, pts_per_ball
    else:
        return group_idx


class Conv1d(nn.Module):
    '''
    - 1D convolution with Batch Normalization
    Note : if_bn for batch normalization layer in the forward pass
    '''
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=1,
                 stride=1,
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class Conv2d(nn.Module):
    '''
    - 2D convolution with Batch Normalization
    Note : if_bn for batch normalization layer in the forward pass
    '''
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP(nn.Module):
    '''
    MLP - Multi-Layer Perceptron with nn.Sequential
    bn -  batch normalization
    '''
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_CONV(nn.Module):
    '''
    MLP_CONV - Multi-Layer Perceptron with 1D Convolutional Neural Network
    '''
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
    '''
    MLP_Res -  Multi-Layer Perceptron with 1D Convolutional Neural Network and
    a shortcut(skip connection ?)
    '''

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


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
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
    new_xyz = gather_operation(xyz,
                               sample_farthest_points(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped,
                     new_xyz.permute(0, 2,
                                     1).contiguous())  # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

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


class PointNet_SA_Module(nn.Module):
    '''
    PointNet_SA_Module -  PointeNet Set of Abstraction layer
    https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf
    https://medium.com/@sanketgujar95/https-medium-com-sanketgujar95-pointnetplus-5d2642560c0d
    '''
    def __init__(self,
                 npoint,
                 nsample,
                 radius,
                 in_channel,
                 mlp,
                 if_bn=True,
                 group_all=False,
                 use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def _sample(self, xyz, npoint):
        """
        Input:
            xyz: input points position data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
        """
        if npoint is None:
            return None

        N = xyz.shape[-1]
        S = absolute_or_relative(npoint, N)  # TODO: Could move the code here, so everything is together

        if N == S:
            new_xyz = xyz  # (B, C, S)
        elif N > S:
            new_xyz = sample_farthest_points(xyz.transpose(-2, -1), K=S, random_start_point=True)[0].transpose(-2, -1)  # (B, C, S)
        else:
            raise RuntimeError(f"can't sample more points than are in point cloud: {N} < {S}")

        return new_xyz

    def _group(self, xyz, points, new_xyz):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            new_xyz: sampled points position data, [B, C, S], or None
        Return:
            grouped_points: grouped points position + feature data, [B, C+D, S, K]
        """
        assert (self.nsample is None) == (new_xyz is None)

        if self.nsample is not None:
            if self.radius is None:
                # KNN
                group_idx = knn_points(new_xyz.transpose(-2, -1), xyz.transpose(-2, -1), K=self.nsample, return_sorted=False).idx  # (B, S, K)
                grouped_xyz = index_points(xyz.transpose(-2, -1), group_idx).movedim(-1, -3)  # (B, C, S, K)
                grouped_xyz -= new_xyz.unsqueeze(-1)  # (B, C, S, 1)
            else:
                # Radius query
                group_idx = query_ball_point(self.radius, self.nsample, xyz.transpose(-2, -1), new_xyz.transpose(-2, -1))  # (B, S, K)
                grouped_xyz = index_points(xyz.transpose(-2, -1), group_idx).movedim(-1, -3)  # (B, C, S, K)
                grouped_xyz -= new_xyz.unsqueeze(-1)  # (B, C, S, 1)
                grouped_xyz /= self.radius

            if points is not None:
                grouped_points = index_points(points.transpose(-2, -1), group_idx).movedim(-1, -3)  # (B, C, S, K)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-3)  # (B, C+D, S, K)
            else:
                grouped_points = grouped_xyz  # (B, C, S, K)
        else:
            # Group all
            grouped_xyz = xyz.unsqueeze(-2)  # (B, C, 1, N)

            if points is not None:
                grouped_points = points.unsqueeze(-2)  # (B, C, 1, N)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-3)  # (B, C+D, 1, K)
            else:
                grouped_points = grouped_xyz  # (B, C, 1, K)

        return grouped_points
    def forward(self, xyz, points=None, new_xyz=None, npoint=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: sample points feature data, [B, D', S]
            new_xyz: sampled points position data, [B, C, S]
        """
        assert (npoint is None) or (self.npoint is None)
        if npoint is None:
            npoint = self.npoint
        assert (npoint is None) == (self.nsample is None)  # both None => group all, otherwise sample and group

        if new_xyz is None:
            new_xyz = self._sample(xyz, npoint)  # (B, C, S)
        grouped_points = self._group(xyz, points, new_xyz)  # (B, C+D, S, K)

        for conv, norm in zip(self.convs, self.norms):
            grouped_points = self.activation(norm(conv(grouped_points)))
        grouped_points = self.last_conv(grouped_points)
        new_points = torch.max(grouped_points, -1)[0]  # (B, D', S)

        return new_xyz, new_points

  # def forward(self, xyz, points):
  #     """
  #     Args:
  #         xyz: Tensor, (B, 3, N)
  #         points: Tensor, (B, f, N)

  #     Returns:
  #         new_xyz: Tensor, (B, 3, npoint)
  #         new_points: Tensor, (B, mlp[-1], npoint)
  #     """
  #     if self.group_all:
  #         new_xyz, new_points, idx, grouped_xyz = self._sample(xyz, points)
  #             #(                xyz, points, self.use_xyz))
  #     else:
  #         new_xyz, new_points, idx, grouped_xyz = sample_and_group(
  #             xyz, points, self.npoint, self.nsample, self.radius,
  #             self.use_xyz)

  #     new_points = self.mlp_conv(new_points)
  #     new_points = torch.max(new_points, 3)[0]

  #     return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    '''
    PointNet feature Propagation layer
    '''
    def __init__(self,
                 in_channel,
                 mlp,
                 use_points1=False,
                 in_channel_points1=None,
                 if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        N = xyz1.shape[-1]
        S = xyz2.shape[-1]
        K = 3

     #   if N <= S:
     #       warnings.warn(f"Fine level doesn't have more points than coarse level: {N} <= {K}")
     #   if S < K:
     #       raise ValueError(f"Passed fewer points (in coarse level) than are required for interpolation (to fine level): {S} < {K}")

        # Flip feature and spatial dimensions
        xyz1_flipped = xyz1.transpose(-2, -1)  # (B, N, C)
        xyz2_flipped = xyz2.transpose(-2, -1)  # (B, N, C)
        points1_flipped = points1.transpose(-2, -1)  # (B, N, D1)
        points2_flipped = points2.transpose(-2, -1)  # (B, N, D2)

        # For each point in the finer level find points in the coarse level for feature interpolation
        dist, idx, _ = knn_points(xyz1_flipped, xyz2_flipped, K=K, return_sorted=False)  # (B, N, K), (B, N, K)
        nn_points_flipped = index_points(points2_flipped, idx)  # (B, N, K, D2)
        nn_points = nn_points_flipped.movedim(-1, -3)   # (B, D2, N, K)

        dist, idx = knn_points(
            xyz1.permute(0, 2, 1).contiguous(),
            xyz2.permute(0, 2, 1).contiguous(),K=3)
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
       #interpolated_points = three_interpolate(points2, idx,
       #                                        weight)  # B, in_channel, N
        dist_recip = 1.0 / (dist + 1e-8)  # (B, N, K)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)  # (B, N, 1)
        weight = dist_recip / norm  # (B, N, K)
        interpolated_points = torch.sum(nn_points * weight.unsqueeze(-3), dim=-1)  # (B, D2, N)

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


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
    new_xyz = gather_operation(xyz,
                               sample_farthest_points(xyz_flipped,
                                                     npoint))  # (B, 3, npoint) wrong
    if idx is None:
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


def fps_subsample(pcd, n_points=2048):#Not so correct !!!
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    if pcd.shape[1] == n_points:
        return pcd
    elif pcd.shape[1] < n_points:
        raise ValueError(
            'FPS subsampling receives a larger n_points: {:d} > {:d}'.format(
                n_points, pcd.shape[1]))
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(),
        sample_farthest_points(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis


def indexing_neighbor(x, index):
    """
    Args:
        x: (bs, dim, num_points0)
        index: (bs, num_points, k)
    Return:
        feature: (bs, dim, num_points, k)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
    feature = x[id_0, index]  # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1,
                              2).contiguous()  # (bs, num_dims, num_points, k)

    return feature


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

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
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