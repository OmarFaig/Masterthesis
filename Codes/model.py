import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import warnings
import logging
from PointNet2 import MLP_Res,MLP_CONV, PointNet_SA_Layer,vTransformer
logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    def __init__(self):
        pass



class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=20, factor=2, attn_channel=True):
        super(SeedGenerator, self).__init__()
        self.uptrans = UpTransformer(256, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel, up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor (B, feat_dim, 1)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        completion = self.mlp_4(x3)  # (B, 3, 256)
        return completion, x3
