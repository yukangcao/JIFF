#### function function modified from Conv-Onet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from .common import coordinate2index, normalize_coordinate, map2local
from .unet3d import UNet3D
from .pointnetpp import PointNetPlusPlus


class PointNetppAUnet3D(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        grid_resolution (int): defined resolution for grid feature
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=32, dim=None, hidden_dim=32, scatter_type='max', grid_resolution=64, padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim
        
        self.dim = dim

        self.pointnetpp = PointNetPlusPlus(dim=self.dim)
        self.unet3d = UNet3D(32, 128, f_maps=32, num_levels=4, layer_order='gcr')
        
        self.reso_grid = grid_resolution
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_grid_features(self, tdmm, c):
    
        tdmm_nor = tdmm.clone()
        index = coordinate2index(tdmm_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(tdmm.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(tdmm.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)
        
        fea_grid = self.unet3d(fea_grid)
                
        return fea_grid


    def forward(self, tdmm):

        # acquire the index for each point
        coord = tdmm.clone()
        index = coordinate2index(coord, self.reso_grid, coord_type='3d')
        
        xyz_pp, c = self.pointnetpp(tdmm)
        
        fea = self.generate_grid_features(tdmm, c)
        return fea
