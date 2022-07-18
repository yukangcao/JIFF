import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNormalizer_fine(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer_fine, self).__init__()
        self.opt = opt

    def forward(self, z, calibs=None, index_feat=None):
        '''
        Normalize z_feature
        :param z_feat: [B, 1, N] depth value for z in the image coordinate system
        :return:
        '''
        z_feat = z * (1024 // 2) / self.opt.z_size
        return z_feat
