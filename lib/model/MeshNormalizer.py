import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np

class MeshNormalizer(nn.Module):
    def __init__(self):
        super(MeshNormalizer, self).__init__()

    def forward(self, tdmm_vertices):
        '''
        Normalize tdmm_vertices
        py_min, py_max are set to be the y_min and y_max of the BBox
        return: normlized tdmm vertices
        '''
        py_min = -28
        py_max = 228
        
        tdmm_vertices_norm = tdmm_vertices.squeeze(1).squeeze(1)
        tdmm_vertices_norm = tdmm_vertices_norm.permute(0, 2, 1)
        tdmm_vertices_norm[:, 1, :] = tdmm_vertices_norm[:, 1, :] - py_min
        tdmm_vertices_norm = tdmm_vertices_norm / (py_max - py_min)
        tdmm_vertices_norm[:, 1, :] = tdmm_vertices_norm[:, 1, :] - 0.5
        
        return py_min, py_max, tdmm_vertices_norm
