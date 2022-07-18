from torch.utils.data import Dataset
import numpy as np
import os
import random
import math
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import time

log = logging.getLogger('trimesh')
log.setLevel(40)

def make_rotate(rx, ry, rz):
    '''
    Used for rotating the 3dmm mesh to certain angle,
    as we just keep the frontal view mesh in the preprocessed dataset
    for better alignment and saving disk storage
    '''
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

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def load_obj_mesh(mesh_file, with_normal=False, with_color=False):
    
    tdmmpt_vertex = []
    tdmmpt_norm = []
    vertex_color_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            # coordinates only: v x y z
            v = list(map(float, values[1:4]))
            tdmmpt_vertex.append(v)
            # coordinates & color: v x y z r g b
            # if len(values) == 7 :
            vc = list(map(float, values[4:7]))
                # Append 1 for Alpha in RGBA
                # vc = list(map(float, values[4:7]+[1]))
            vertex_color_data.append(vc)
            
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            tdmmpt_norm.append(vn)

    vertices = np.array(tdmmpt_vertex)
    vertex_color = np.array(vertex_color_data)
    
    if with_normal:
        norms = np.array(tdmmpt_norm)
        norms = normalize_v3(norms)
        return vertices, norms
    if with_color:
        return vertices, vertex_color

        
    return vertices

class TrainDataset_jiff(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        
        self.FACE_3DMM = os.path.join(self.root, 'FACE_3DMM')
        self.FACE_REGION = os.path.join(self.root, 'FACE_REGION')
        self.FILE_NUMBER = os.path.join(self.root, 'FILE_NUMBER')
        
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        
        ## BBox for samples randomly in image space
        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])
        
        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize
        self.num_views = self.opt.num_views
        
        # number of sample points
        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.to_tensor_1024 = transforms.Compose([
            transforms.Resize(1024),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        # NOTE: we choose to load trimesh during the training procedure
#        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = os.listdir(self.UV_RENDER)
        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render_shape(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'face_region': [num_views, 1, 4] region to locate face in 3D space
            'face_tdmm': [num_views, n, 3] 3dmm vertex points
            'view_id': [num_views, 1, 1] view angle for rotating the 3D points
        '''
        pitch = self.pitch_list[pid]
        
        # For our training, we just used the images with detected face
        # You can uncomment this part if you want to train with 360-degree.
        
#        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
#                    for offset in range(num_views)]
#        if random_sample:
#            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
        
        # The ids are an even distribution of num_views around view_id
        view_number_path = os.path.join(self.FILE_NUMBER, subject, 'number.txt')
        fp = open(view_number_path, 'r')
        lines= fp.readlines()
        fp.close()
        view_ids = []
        for line in lines:
            line = int(line.replace('\n', ''))
            view_ids.append(line)
       
        view_ids = [view_ids[(yid + len(view_ids) // num_views * offset) % len(view_ids)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
        
        calib_list = []
        render_list = []
        
        face_tdmm_list = []
        face_region_list = []
        
        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render
            render_list.append(render)
            calib_list.append(calib)

            # face region
            faceRegion_path = os.path.join(self.FACE_REGION, subject, '%d_%d_%02d.txt' % (0, 0, 0))
            
            fp = open(faceRegion_path, 'r')
            lines = fp.readlines()
            fp.close()
            FaceRegion = []
            for line in lines:
                line_data = line.strip().split(', ')
                FaceRegion.append([float(line_data[0]), float(line_data[1]), float(line_data[2]), float(line_data[3])])
                continue
            Top = FaceRegion[0][0]
            Bottom = FaceRegion[0][1]
            Left = FaceRegion[0][2]
            Right = FaceRegion[0][3]
            
            FaceRegion_new = []
            FaceRegion_new.append(Top); FaceRegion_new.append(Bottom)
            FaceRegion_new.append(Left); FaceRegion_new.append(Right)
            FaceRegion_new = torch.tensor(FaceRegion_new).float()
            face_region_list.append(FaceRegion_new)
            
            # tdmm
            tdmm_path = os.path.join(self.FACE_3DMM, subject, '%d_%d_%02d_mesh.obj' % (0, 0, 0))
            tdmm_vertex = load_obj_mesh(tdmm_path)
            tdmm_vertex = torch.from_numpy(tdmm_vertex).permute(1, 0)
                        
            angle = int(vid)
            R = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(angle), 0))
            tdmm_vertex = np.dot(R, tdmm_vertex)
            tdmm_vertex = torch.from_numpy(tdmm_vertex).permute(1, 0).unsqueeze(0).float()
            
            face_tdmm_list.append(tdmm_vertex)
            
            del tdmm_vertex
        view_id = []
        view_id.append(torch.from_numpy(np.array(view_ids)))
        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'face_region': torch.stack(face_region_list, dim=0),
            'face_tdmm': torch.stack(face_tdmm_list, dim=0),
            'view_id': torch.stack(view_id, dim=0)
        }

    def get_render_color(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img_1024': [num_views, C, W, H] high-res images
            'img_512': [num_views, C, W, H] low-res images
            'calib_1024': [num_views, 4, 4] calibration matrix for high-res images
            'calib_512': [num_views, 4, 4] calibration matrix for high-res images
            'face_region': [num_views, 1, 4] region to locate face in 3D space
            'face_tdmm': [num_views, n, 6] 3dmm vertex points with vertex color
            'view_id': [num_views, 1, 1] view angle for rotating the 3D points
        '''
        pitch = self.pitch_list[pid]

        # The ids are an even distribution of num_views around view_id
        view_number_path = os.path.join(self.FILE_NUMBER, subject, 'number.txt')
        fp = open(view_number_path, 'r')
        lines= fp.readlines()
        fp.close()
        view_ids = []
        for line in lines:
            line = int(line.replace('\n', ''))
            view_ids.append(line)
        
        view_ids = [view_ids[(yid + len(view_ids) // num_views * offset) % len(view_ids)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        calib_1024_list = []
        render_1024_list = []
        
        render_512_list = []
        calib_512_list = []
                
        face_tdmm_list = []
        face_region_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)


                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))

                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            
            calib_1024 = calib
            mask_1024 = mask
            render_1024 = render
            
            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
                        
            mask_1024 = transforms.Resize(1024)(mask_1024)
            mask_1024 = transforms.ToTensor()(mask_1024).float()

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render
            
            render_1024 = self.to_tensor_1024(render_1024)
            render_1024 = mask_1024.expand_as(render_1024) * render_1024
            
            render_1024_list.append(render_1024)
            render_512_list.append(render)
            calib_1024_list.append(calib_1024)
            calib_512_list.append(calib)

            # face region
            faceRegion_path = os.path.join(self.FACE_REGION, subject, '%d_%d_%02d.txt' % (0, 0, 0))
            
            fp = open(faceRegion_path, 'r')
            lines = fp.readlines()
            fp.close()
            FaceRegion = []
            for line in lines:
                line_data = line.strip().split(', ')
                FaceRegion.append([float(line_data[0]), float(line_data[1]), float(line_data[2]), float(line_data[3])])
                continue
            Top = FaceRegion[0][0]
            Bottom = FaceRegion[0][1]
            Left = FaceRegion[0][2]
            Right = FaceRegion[0][3]
            
            FaceRegion_new = []
            FaceRegion_new.append(Top); FaceRegion_new.append(Bottom)
            FaceRegion_new.append(Left); FaceRegion_new.append(Right)
            FaceRegion_new = torch.tensor(FaceRegion_new).float()
            face_region_list.append(FaceRegion_new)
            
            # tdmm
            tdmm_path = os.path.join(self.FACE_3DMM, subject, '%d_%d_%02d_mesh.obj' % (0, 0, 0))
            tdmm_vertex, tdmm_color = load_obj_mesh(tdmm_path, with_color=True)
            tdmm_vertex = torch.from_numpy(tdmm_vertex).permute(1, 0)
                        
            angle = int(vid)
            R = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(angle), 0))
            tdmm_vertex = np.dot(R, tdmm_vertex)
            tdmm_vertex = torch.from_numpy(tdmm_vertex).permute(1, 0).unsqueeze(0).float()
            tdmm_color = torch.from_numpy(tdmm_color).unsqueeze(0)
            tdmm_vertex = torch.cat([tdmm_vertex, tdmm_color], -1)
            
            face_tdmm_list.append(tdmm_vertex)
            del tdmm_vertex
                    
        view_id = []
        view_id.append(torch.from_numpy(np.array(view_ids)))
        return {
            'img_1024': torch.stack(render_1024_list, dim=0),
            'img_512': torch.stack(render_512_list, dim=0),
            'calib_1024': torch.stack(calib_1024_list, dim=0),
            'calib_512': torch.stack(calib_512_list, dim=0),
            'face_region': torch.stack(face_region_list, dim=0),
            'face_tdmm': torch.stack(face_tdmm_list, dim=0),
            'view_id': torch.stack(view_id, dim=0)
        }
    def get_shape_sampling(self, subject):
    
#        mesh = self.mesh_dic[subject]
        mesh = trimesh.load(os.path.join(self.OBJ, subject, subject + '.obj'))
        mesh_vertices = mesh.vertices
        mesh_faces = mesh.faces
        z_range_list = []
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)
        
        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        
        sample_points = np.concatenate([sample_points, random_points], 0).astype(float)
        
        np.random.shuffle(sample_points)
                               
        inside = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces).contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]
        
        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
                
        ###### add random points within face region
        # locate the face region
        faceRegion_path = os.path.join(self.FACE_REGION, subject, '%d_%d_%02d.txt' % (0, 0, 0))

        fp = open(faceRegion_path, 'r')
        lines = fp.readlines()
        fp.close()
        FaceRegion = []
        for line in lines:
            line_data = line.strip().split(', ')
            FaceRegion.append([float(line_data[0]), float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]), float(line_data[5])])
            continue
        ### 3d face region
        # xy
        Top = float((1.0 - FaceRegion[0][0] / 256.0) * 90 + 100) # ymax
        Bottom = float((1.0 - FaceRegion[0][1] / 256.0) * 90 + 100) # ymin
        Left = float(FaceRegion[0][2] / 2.0 - 128) # xmin
        Right = float(FaceRegion[0][3] / 2.0 - 128) # xmax
        # z
        zmin = float(FaceRegion[0][4])
        zmax = float(FaceRegion[0][5])

        ### face box in image space
        F_MIN = np.array([Left, Bottom, zmin])
        F_MAX = np.array([Right, Top, zmax])
        F_length = F_MAX - F_MIN
        
        ### face sampled points and labels
        face_points = np.random.rand(700, 3) * F_length + F_MIN

        face_inside = mesh.contains(face_points)
        face_inside_points = face_points[face_inside]
        face_outside_points = face_points[np.logical_not(face_inside)]
        
        face_samples = np.concatenate([face_inside_points, face_outside_points], 0).T

        face_labels = np.concatenate([np.ones((1, face_inside_points.shape[0])), np.zeros((1, face_outside_points.shape[0]))], 1)
                
        ##### combination
        samples = np.concatenate([samples, face_samples], 1)
        labels = np.concatenate([labels, face_labels], 1)
                
        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        return {
            'samples': samples,
            'labels': labels
        }

    
    def get_color_sampling(self, subject, yid, pid=0):
    
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1)) # (loadSize * loadSize, )
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))
        
        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]
        
        # face region
        faceRegion_path = os.path.join(self.FACE_REGION, subject, '%d_%d_%02d.txt' % (0, 0, 0))

        fp = open(faceRegion_path, 'r')
        lines = fp.readlines()
        fp.close()
        FaceRegion = []
        for line in lines:
            line_data = line.strip().split(', ')
            FaceRegion.append([float(line_data[0]), float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]), float(line_data[5])])
            continue
        ### 3d face region
        # xy
        Top = float((1.0 - FaceRegion[0][0] / 256.0) * 90 + 100) # ymax
        Bottom = float((1.0 - FaceRegion[0][1] / 256.0) * 90 + 100) # ymin
        Left = float(FaceRegion[0][2] / 2.0 - 128) # xmin
        Right = float(FaceRegion[0][3] / 2.0 - 128) # xmax
        
        in_face = (surface_points[:, 0] >= Left) & (surface_points[:, 0] <= Right) & (surface_points[:, 1] >= Bottom) & (surface_points[:, 1] <= Top)
        face_points = surface_points[in_face, :]
        face_colors = surface_colors[in_face, :]
        face_normal = surface_normal[in_face, :]
        
        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            face_list = random.sample(range(0, face_points.shape[0] - 1), 700)
            
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T
            
            face_points = face_points[face_list].T
            face_colors = face_colors[face_list].T
            face_normal = face_normal[face_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal
                  
        face_normal = torch.Tensor(face_normal).float()
        face_samples = torch.Tensor(face_points).float() \
                  + torch.normal(mean=torch.zeros((1, face_normal.size(1))), std=self.opt.sigma).expand_as(face_normal) * face_normal
        
        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0
        rgbs_face_color = 2.0 * torch.Tensor(face_colors).float() - 1.0
        
        samples = torch.cat([samples, face_samples], 1)
        rgbs = torch.cat([rgbs_color, rgbs_face_color], 1)
        
        return {
            'color_samples': samples,
            'rgbs': rgbs
        }


    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject+'.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }

        if self.opt.num_sample_inout:
            render_data = self.get_render_shape(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.opt.random_multiview)
            res.update(render_data)
            sample_data = self.get_shape_sampling(subject)
            res.update(sample_data)

        if self.num_sample_color:
            render_data = self.get_render_color(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.opt.random_multiview)
            res.update(render_data)
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res

    def __getitem__(self, index):
#        import cv2
#        cv2.setNumThreads(0)
        return self.get_item(index)
