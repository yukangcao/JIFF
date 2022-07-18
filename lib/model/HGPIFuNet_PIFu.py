import torch
import torch.nn as nn
import numpy as np
import trimesh
import torch.nn.functional as F
import math
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier, SurfaceClassifier_joint
from .DepthNormalizer import DepthNormalizer
from .MeshNormalizer import MeshNormalizer
from .HGFilters import *
from .PointNetFilters import LocalPoolPointnet
from .PointNetFilters_pp import PointNetppAUnet3D
from ..net_util import init_net
from .common import normalize_3d_coordinate

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R
    
class HGPIFuNet_PIFu(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet_PIFu, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)
        self.tdmm_filter = LocalPoolPointnet()
        self.smpl_filter = LocalPoolPointnet()
        
        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())
        
        self.surface_classifier_multiLoss = SurfaceClassifier_joint(opt=opt,
            filter_channels_2d=self.opt.mlp_dim,
            filter_channels_3d=self.opt.mlp_dim_3d,
            filter_channels_joint=self.opt.mlp_dim_joint,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)
        self.MeshNormalizer = MeshNormalizer()

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.im_tdmm_feat_list = []
        self.tmpx = None
        self.normx = None
        
        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        # print(self.im_feat_list[-1].cpu().detach().numpy().shape)
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def filter_tdmm(self, face_tdmm):
        '''
        Filter the input tdmm mesh
        '''
        self.im_tdmm_grid_feats = self.tdmm_filter(face_tdmm)

    def query(self, points, calibs, transforms=None, labels=None, face_region=None, py_min=None, py_max=None, view_id=None, sdf=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :param face_region: [B, 1, 4] face region in 2d space
        :param py_min & py_max: [1] value for rescaling the sample points to [-0.5,0.5]
        :param view_id: angle for rotating sample points to index 3D features
        
        :return: [B, Res, N] predictions for each point
        '''
        
        batch_size = points.size()[0]
        
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        z_feat = self.normalizer(z, calibs=calibs)
        
        # list for storing predictions
        self.intermediate_preds_list = []
        self.intermediate_preds_face_list = []
            
        points_xy = points[:, :2, :]
        points_z = points[:, 2:3, :]
        
        # get the labels for face pointss
        if labels is not None:
            self.labels = labels
            self.labels_face = self.labels[:, :, in_mesh[0, :]]
        
        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]
    
    def get_im_tdmm_feat(self):
        '''
        Get the 3dmm mesh filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_tdmm_grid_feats
        
    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        for preds_face in self.intermediate_preds_face_list:
            error += self.error_term(preds_face, self.labels_face)
        error /= (len(self.intermediate_preds_list) + len(self.intermediate_preds_face_list))
        
        return error

    def forward(self, images, points, calibs, tdmm_vertex=None, face_region=None, transforms=None, labels=None, view_id=None):
    
        # Phase 1: get the 2D and 3D features
        # Get image feature
        self.filter(images)
        
        # Get the 3dmm feature
        if tdmm_vertex != None:
            py_min, py_max, tdmm_vertices_norm = self.MeshNormalizer(tdmm_vertex)
            tdmm_vertex = tdmm_vertices_norm.squeeze(0).permute(1, 0)
            tdmm_vertex_norm = normalize_3d_coordinate(tdmm_vertex, padding=0.1)
            
            tdmm_vertex_norm = tdmm_vertex_norm.unsqueeze(0)
            self.filter_tdmm(tdmm_vertex_norm)
            
        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels, face_region=face_region, py_min=py_min, py_max=py_max, view_id=view_id)
        
        # get the prediction
        res = self.get_preds()
            
        # get the error
        error = self.get_error()

        return res, error
