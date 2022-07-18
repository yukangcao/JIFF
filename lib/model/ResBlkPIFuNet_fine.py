import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BasePIFuNet import BasePIFuNet
from .ResBlkPIFuNet import ResBlkPIFuNet
from .HGPIFuNet import HGPIFuNet
import functools
from .SurfaceClassifier import SurfaceClassifier, SurfaceClassifier_joint
from .DepthNormalizer_fine import DepthNormalizer_fine
from .MeshNormalizer import MeshNormalizer
from .PointNetFilters_pp import PointNetppAUnet3D
from ..net_util import *
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
    
class ResBlkPIFuNet_fine(BasePIFuNet):
    def __init__(self, opt,
                 projection_mode='orthogonal'):
        if opt.color_loss_type == 'l1':
            error_term = nn.L1Loss()
        elif opt.color_loss_type == 'mse':
            error_term = nn.MSELoss()

        super(ResBlkPIFuNet_fine, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'respifu'
        self.opt = opt

        norm_type = get_norm_layer(norm_type=opt.norm_color)
        self.image_filter = ResnetFilter(opt, norm_layer=norm_type)
        self.tdmm_filter = PointNetppAUnet3D(dim=6)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_color_fine,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Tanh())
            
        self.surface_classifier_joint = SurfaceClassifier_joint(opt=opt,
            filter_channels_2d=self.opt.mlp_dim_color_fine,
            filter_channels_3d=self.opt.mlp_dim_3d_color,
            filter_channels_joint=self.opt.mlp_dim_joint_color,
            last_op=nn.Tanh())

        self.normalizer = DepthNormalizer_fine(opt)
        self.MeshNormalizer = MeshNormalizer()
        
        self.intermediate_preds_list = []
        self.intermediate_preds_face_list = []
        
        init_net(self)


    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat = self.image_filter(images)
        
    def filter_tdmm(self, face_tdmm):
        '''
        Filter the input tdmm mesh
        store all intermediate features.
        '''
        self.im_tdmm_grid_feats = self.tdmm_filter(face_tdmm)

    def attach(self, im_feat):
        self.im_feat = torch.cat([im_feat, self.im_feat], 1)

    def query(self, points, calibs, coarse_mlp_features, transforms=None, labels=None, face_region=None, py_min=None, py_max=None, view_id=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        Region = face_region.detach().cpu().numpy()
        
        if Region.shape == (1, 1, 4):
            Region = np.squeeze(Region, axis=(0,))
        
        batch_size = points.size()[0]
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        z_feat = self.normalizer(z)
        
        self.intermediate_preds_list = []
        self.intermediate_preds_face_list = []
            
        face_ymin = (1.0 - Region[0][1] / 256.0) * 90 + 100
        face_ymax = (1.0 - Region[0][0] / 256.0) * 90 + 100
        face_xmin = (Region[0][2] / 2.0 - 128) / 1.0
        face_xmax = (Region[0][3] / 2.0 - 128) / 1.0
                    
        points_xy = points[:, :2, :]
        points_z = points[:, 2:3, :]
        
        in_mesh = (points_xy[:, 0] >= face_xmin) & (points_xy[:, 0] <= face_xmax) & (points_xy[:, 1] >= face_ymin) & (points_xy[:, 1] <= face_ymax)
        out_mesh = ~in_mesh
        
        if labels is not None:
            self.labels = labels
            self.labels_face = self.labels[:, :, in_mesh[0, :]]
        
        points_scale = points.clone()
        points_scale[:, 1, :] = points_scale[:, 1, :] - py_min
        points_scale = points_scale / (py_max - py_min)
        points_scale[:, 1, :] = points_scale[:, 1, :] - 0.5
                    
        if view_id != None:
            angle = int(view_id)
            R = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(angle), 0))
            rotate_vertex = points_scale.squeeze(0).cpu()
            rotate_vertex = np.dot(R, rotate_vertex)
            points_scale = torch.from_numpy(rotate_vertex).unsqueeze(0).float().to(device=points.device)
                            
        points_inface = points_scale
        points_inface_norm = normalize_3d_coordinate(points_inface, padding=0.1)
                
        if np.sum(in_mesh.float().detach().cpu().numpy()) != 0:
        
            points_num = int(np.sum(in_mesh.float().detach().cpu().numpy()))
            # This is a list of [B, Feat_i, N] features
            point_local_feat_list = [self.index(self.im_feat, xy), coarse_mlp_features, z_feat]
            # [B, Feat_all, N]
            point_local_feat = torch.cat(point_local_feat_list, 1)
            
            # add 3d tdmm feature
            tdmm_feats = self.im_tdmm_grid_feats
            points_inface_norm_grid = points_inface_norm[:, :, in_mesh[0, :]]
            points_save = points_inface_norm_grid.cpu().squeeze(0).permute(1, 0)

            points_inface_norm_grid[:, 0, :] -= 0.5
            points_inface_norm_grid[:, 1, :] -= 0.5
            points_inface_norm_grid[:, 2, :] -= 0.5

            points_inface_norm_grid[:, 0, :] *= 2
            points_inface_norm_grid[:, 1, :] *= 2
            points_inface_norm_grid[:, 2, :] *= 2

            interp_tdmm_3dfeats = F.grid_sample(tdmm_feats, points_inface_norm_grid.permute(0, 2, 1).unsqueeze(2).unsqueeze(2), mode='bilinear')

            inmesh_local_feat3d = interp_tdmm_3dfeats.view([batch_size, -1, points_num])
            
            # [B, Feat_all, N]
            point_local_feat = torch.cat(point_local_feat_list, 1)
                        
            inmesh_local_feat2d = point_local_feat[:, :, in_mesh[0, :]]
            
            pred = self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)
                
            pred_face_list = self.surface_classifier_joint(feature_2d=inmesh_local_feat2d, feature_3d=inmesh_local_feat3d)
                            
            for pred_face in pred_face_list:
                self.intermediate_preds_face_list.append(pred_face)
                
        else:
            point_local_feat_list = [self.index(self.im_feat, xy), coarse_mlp_features, z_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred = self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)
            
        self.preds = self.intermediate_preds_list[-1]
        
        if np.sum(in_mesh.float().detach().cpu().numpy()) != 0 and not self.training:
            self.preds[:, :, in_mesh[0, :]] = self.intermediate_preds_face_list[-1]
            
    def get_error(self):
    
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        for preds_face in self.intermediate_preds_face_list:
            error += self.error_term(preds_face, self.labels_face)
        error /= (len(self.intermediate_preds_list) + len(self.intermediate_preds_face_list))
        
        return error
        
    def forward(self, images_1024, coarse_feat, points, calibs_1024, face_region=None, tdmm_vertex=None, view_id=None, transforms=None, labels=None):
    
        # get coarse mlp intermediate output from coarse pipeline
        
        self.filter(images_1024)
        tdmm_vertices = tdmm_vertex[:, :, :, :, :3]
        tdmm_color = tdmm_vertex[:, :, :, :, 3:].squeeze(0).squeeze(0).squeeze(0)
        
        if tdmm_vertex != None:
            py_min, py_max, tdmm_vertices_norm = self.MeshNormalizer(tdmm_vertices)
            tdmm_vertices = tdmm_vertices_norm.squeeze(0).permute(1, 0)
            tdmm_vertices_norm = normalize_3d_coordinate(tdmm_vertices, padding=0.1)
        tdmm_vertice_norm = tdmm_vertices_norm.unsqueeze(0)
        
        tdmm_vertex = torch.cat([tdmm_vertices_norm, tdmm_color], 1)
        tdmm_vertex = tdmm_vertex.unsqueeze(0).float()
        self.filter_tdmm(tdmm_vertex)
        
        self.query(points, calibs_1024, coarse_feat, transforms, labels, face_region=face_region, py_min=py_min, py_max=py_max, view_id=view_id)

        res = self.get_preds()
        error = self.get_error()

        return res, error

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, last=True)]
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]

        if opt.use_tanh:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
