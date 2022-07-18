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
    
class HGPIFuNet(BasePIFuNet):
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
        super(HGPIFuNet, self).__init__(
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
        :param sdf: sdf calculated between 3dmm mesh and query points, for fusing mlps
        
        :return: [B, Res, N] predictions for each point
        '''
        
        Region = face_region.detach().cpu().numpy()
        
        if Region.shape == (1, 1, 4):
            Region = np.squeeze(Region, axis=(0,))
        
        batch_size = points.size()[0]
        
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        z_feat = self.normalizer(z, calibs=calibs)
        
        # list for storing predictions
        self.intermediate_preds_list = []
        self.intermediate_preds_face_list = []
            
        # transforming face region from 2D to 3D
        # no need to restrict the z axis
        face_ymin = (1.0 - Region[0][1] / 256.0) * 90 + 100
        face_ymax = (1.0 - Region[0][0] / 256.0) * 90 + 100
        face_xmin = (Region[0][2] / 2.0 - 128) / 1.0
        face_xmax = (Region[0][3] / 2.0 - 128) / 1.0
            
        points_xy = points[:, :2, :]
        points_z = points[:, 2:3, :]
        
        # mask for determing if points lie in face region
        in_mesh = (points_xy[:, 0] >= face_xmin) & (points_xy[:, 0] <= face_xmax) & (points_xy[:, 1] >= face_ymin) & (points_xy[:, 1] <= face_ymax)
        out_mesh = ~in_mesh
        
        # get the labels for face pointss
        if labels is not None:
            self.labels = labels
            self.labels_face = self.labels[:, :, in_mesh[0, :]]
        
        # normalize the sample points to [-0.5, 0.5]
        points_scale = points.clone()
        points_scale[:, 1, :] = points_scale[:, 1, :] - py_min
        points_scale = points_scale / (py_max - py_min)
        points_scale[:, 1, :] = points_scale[:, 1, :] - 0.5
        
        # rotate the points to certain angle
        # we won't rotate points when testing
        if view_id != None:
            angle = int(view_id)
            R = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(angle), 0))
            rotate_vertex = points_scale.squeeze(0).cpu()
            rotate_vertex = np.dot(R, rotate_vertex)
            points_scale = torch.from_numpy(rotate_vertex).unsqueeze(0).float().to(device=points.device)
        
        points_inface = points_scale
        points_inface_norm = normalize_3d_coordinate(points_inface, padding=0.1)
        
        # number of face points
        points_num = int(np.sum(in_mesh.float().detach().cpu().numpy()))        
        
        if points_num != 0:

            for i, im_feat in enumerate(self.im_feat_list):
                # original 2d features
                point_local_feat_list = [self.index(im_feat, xy), z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # add 3d tdmm feature
                tdmm_feats = self.im_tdmm_grid_feats

                points_inface_norm_grid = points_inface_norm[:, :, in_mesh[0, :]]
                points_save = points_inface_norm_grid.cpu()[0, :, :].permute(1, 0)

                points_inface_norm_grid[:, 0, :] -= 0.5
                points_inface_norm_grid[:, 1, :] -= 0.5
                points_inface_norm_grid[:, 2, :] -= 0.5

                points_inface_norm_grid[:, 0, :] *= 2
                points_inface_norm_grid[:, 1, :] *= 2
                points_inface_norm_grid[:, 2, :] *= 2

                interp_tdmm_3dfeats = F.grid_sample(tdmm_feats, points_inface_norm_grid.permute(0, 2, 1).unsqueeze(2).unsqueeze(2), mode='bilinear')
                   
                # face-related 2D and 3D features
                inmesh_local_feat3d = interp_tdmm_3dfeats.view([batch_size, -1, points_num])
                
                inmesh_local_feat2d = point_local_feat[:, :, in_mesh[0, :]]
                
                pred = in_img[:, None].float() * self.surface_classifier(point_local_feat)
                self.intermediate_preds_list.append(pred)
                
                pred_face_list = self.surface_classifier_multiLoss(feature_2d=inmesh_local_feat2d, feature_3d=inmesh_local_feat3d)

                for pred_face in pred_face_list:
                    self.intermediate_preds_face_list.append(pred_face)
                
        else:
            for im_feat in self.im_feat_list:
                # [B, Feat_i + z, N]
                point_local_feat_list = [self.index(im_feat, xy), z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # out of image plane is always set to 0
                pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
                self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]
        
        if points_num != 0 and not self.training and sdf != None:
            
            ## if you want, here we apply sdf for mlp fusion
            ## this will:
            ## 1. eliminate the backside 'outliers' along the boundary between head and body
            ## 2. make us feel free to extend the face bbox
            ## 3. decrease the level of head details a little bit at the same time
            ## meanwhile we find that mostly it will just help us dealing the cases with long hair, clothes with hats, etc.
            sdf_face = sdf[:, in_mesh[0, :], :].cpu().numpy()
            
            pred_body_face = self.preds[:, :, in_mesh[0, :]]

            pred_face_face = self.intermediate_preds_face_list[-1]
            
            w = torch.from_numpy(np.exp(-0.03 * sdf_face**2)).to(device=sdf.device).permute(0, 2, 1)
            
            self.preds[:, :, in_mesh[0, :]] = w * pred_face_face + (1-w) * pred_body_face
            
            ## if you don't want to use the mlp fusion strategy
            ## comment the code above and use this line instead
            
#            self.preds[:, :, in_mesh[0, :]] = pred_face_face

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
