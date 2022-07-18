#### function modified from Conv-Onet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from .common import coordinate2index, normalize_coordinate, map2local
from .unet3d import UNet3D


class LocalPoolPointnet(nn.Module):
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

    def __init__(self, c_dim=32, dim=3, hidden_dim=32, scatter_type='max', grid_resolution=64, plane_type='grid', padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim
        
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.fc_pos = nn.Linear(self.dim, 2*self.hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*self.hidden_dim, self.hidden_dim) for i in range(n_blocks)
        ])
        
        self.fc_c = nn.Linear(hidden_dim, self.c_dim)

        self.actvn = nn.ReLU()

        self.unet3d = UNet3D(32, 128, f_maps=32, num_levels=4, layer_order='gcr')

        self.reso_grid = grid_resolution
        self.plane_type = plane_type
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

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)

        c_out = 0
            # scatter plane features from points
        fea = self.scatter(c.permute(0, 2, 1), index, dim_size=self.reso_grid**3)
        if self.scatter == scatter_max:
            fea = fea[0]
            # gather feature back to points
        fea = fea.gather(dim=2, index=index.expand(-1, fea_dim, -1))
        c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, tdmm):

        # acquire the index for each point
        
        coord = tdmm.clone()
        index = coordinate2index(coord, self.reso_grid, coord_type='3d')
        
        net = self.fc_pos(tdmm)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        
        fea = self.generate_grid_features(tdmm, c)
        return fea
