import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceClassifier_joint(nn.Module):
    # modified from Geo-PIFu
    
    def __init__(self, opt, filter_channels_2d, filter_channels_3d, filter_channels_joint, last_op=None):

        super(SurfaceClassifier_joint, self).__init__()

        # ----- 2d features branch -----
        self.filters_2d = []
        for idx2d in range(0, len(filter_channels_2d) - 1):
            if idx2d == 0:
                self.filters_2d.append(
                    nn.Conv1d(filter_channels_2d[idx2d],filter_channels_2d[idx2d + 1], 1))
            else:
                self.filters_2d.append(
                    nn.Conv1d(filter_channels_2d[0] + filter_channels_2d[idx2d], filter_channels_2d[idx2d + 1], 1))
            self.add_module("features_2d_conv%d"%(idx2d), self.filters_2d[idx2d])

        # ----- 3d features branch -----
        self.filters_3d = []
        for idx3d in range(0, len(filter_channels_3d) - 1):
            if idx3d == 0:
                self.filters_3d.append(
                    nn.Conv1d(filter_channels_3d[idx3d], filter_channels_3d[idx3d + 1], 1))
            else:
                self.filters_3d.append(  nn.Conv1d(filter_channels_3d[0] + filter_channels_3d[idx3d], filter_channels_3d[idx3d + 1], 1)  )
            self.add_module("features_3d_conv%d"%(idx3d), self.filters_3d[idx3d])

        # ----- fused features branch -----
        filter_channels_joint[0] = filter_channels_2d[0]  + filter_channels_3d[0]
        filter_channels_fused    = filter_channels_2d[-2] + filter_channels_3d[-2]
        self.filters_joint = []
        for idx in range(0, len(filter_channels_joint) - 1):
            if idx == 0:
                self.filters_joint.append(  nn.Conv1d(filter_channels_joint[0]+     filter_channels_fused, filter_channels_joint[idx + 1], 1)  )
            else:
                self.filters_joint.append(  nn.Conv1d(filter_channels_joint[0] + filter_channels_joint[idx], filter_channels_joint[idx + 1], 1)  )
            self.add_module("features_joint_conv%d"%(idx), self.filters_joint[idx])

        # ----- the last layer for (0., 1.) sdf pred -----
        self.last_layer = last_op

    def forward(self, feature_2d, feature_3d):

#        # init.
        pred = []

        # ----- 2d features branch -----
        feature_2d_skip = feature_2d  ## tmpy
        feature_2d_pass = feature_2d  ## y
        
        for idx in range(len(self.filters_2d)):

            if (idx == len(self.filters_2d) - 1) and (not self.training):
                continue

            feature_2d_pass = self._modules["features_2d_conv%d"%(idx)](feature_2d_pass if idx==0 else torch.cat([feature_2d_pass,feature_2d_skip], 1))
            if idx != len(self.filters_2d) - 1:
                feature_2d_pass = F.leaky_relu(feature_2d_pass)
                if idx == len(self.filters_2d) - 2:
                    feature_2d_fuse = feature_2d_pass
            else:
                pred_2d = self.last_layer(feature_2d_pass)
                pred.append(pred_2d)
                
        # ----- 3d features branch -----
        feature_3d_skip = feature_3d
        feature_3d_pass = feature_3d
        
        for idx in range(len(self.filters_3d)):

            if (idx == len(self.filters_3d) - 1) and (not self.training):
                continue

            feature_3d_pass = self._modules["features_3d_conv%d"%(idx)](feature_3d_pass if idx==0 else torch.cat([feature_3d_pass, feature_3d_skip], 1))
            if idx != len(self.filters_3d) - 1:
                feature_3d_pass = F.leaky_relu(feature_3d_pass)
                if idx == len(self.filters_3d) - 2:
                    feature_3d_fuse = feature_3d_pass
            else:
                pred_3d = self.last_layer(feature_3d_pass)
                pred.append(pred_3d)
                
        # ----- fused features branch -----
        feature_joint_skip = torch.cat([feature_2d_skip, feature_3d_skip], 1)
        feature_joint_pass = torch.cat([feature_2d_fuse, feature_3d_fuse], 1)
        
        for idx in range(len(self.filters_joint)):

            feature_joint_pass = self._modules["features_joint_conv%d"%(idx)](torch.cat([feature_joint_pass,feature_joint_skip], 1))
            if idx != len(self.filters_joint)-1:
                feature_joint_pass = F.leaky_relu(feature_joint_pass)
            else:
                pred_joint = self.last_layer(feature_joint_pass)
                pred.append(pred_joint)
        # return
        return pred

class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()
        # [257, 1024, 512, 256, 128, 1]
        self.filters = [] # 5
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)
        return y

class SurfaceClassifier_color(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier_color, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)
            if i == 2:
                phi = y.clone()

        if self.last_op:
            y = self.last_op(y)

        return y, phi
