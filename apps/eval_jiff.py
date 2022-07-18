import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

def load_obj_mesh(mesh_file, with_normal=False, with_color=False):
    '''
    load 3dmm mesh vertices (and vertex color for color inference)
    '''
    
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
            if len(values) == 7 :
                vc = list(map(float, values[4:7]))
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

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        
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
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        print('loading for net C ...', opt.load_netC_checkpoint_path)
        netC = ResBlkPIFuNet_fine(opt).to(device=cuda)
        netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        netC_coarse = ResBlkPIFuNet(opt).to(device=cuda)
        netC_coarse.load_state_dict(torch.load(opt.load_netC_coarse_checkpoint_path, map_location=cuda))
        self.with_color = True

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC
        self.netC_coarse = netC_coarse

    def load_image(self, image_path, mask_path, region_path, tdmm_path):
        ##### Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        ##### BBox
        # PIFu set the BBox to be [-1, 1] originally
        # We sepecifically set the BBox to help easily align huamn mesh and 3DMM
        B_MIN = np.array([-102.9, -12.9, -101.4])
        B_MAX = np.array([102.9, 192.9, 104.4])
        
        ##### Calibration (Intrinsic)
        projection_matrix = np.identity(4) / 102.9
        projection_matrix[1, 1] = -1 / 102.9
        projection_matrix[1, 3] = 90 / 102.9
        projection_matrix[2, 3] = 1.5 / 102.9
        calib = torch.Tensor(projection_matrix).float()
        calib_1024 = calib * 2.0
        
        ##### Mask
        mask = Image.open(mask_path).convert('L')
        mask_1024 = mask.resize((1024, 1024),Image.ANTIALIAS)
        
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        mask_1024 = transforms.Resize(1024)(mask_1024)
        mask_1024 = transforms.ToTensor()(mask_1024).float()
        
        ##### image
        image = Image.open(image_path).convert('RGB')
        image_1024 = image.resize((1024, 1024),Image.ANTIALIAS)
        
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        image_1024 = self.to_tensor_1024(image_1024)
        image_1024 = mask_1024.expand_as(image_1024) * image_1024
        
        ##### face region
        # We require face region to locate face-related points in 3D space
        # The face_region is detected by mtcnn
        FaceRegion_new = []
        faceRegion_path = image_path.replace('.jpg', '_region.txt')
        fp = open(faceRegion_path, 'r')
        lines = fp.readlines()
        fp.close()
        FaceRegion = []
        for line in lines:
            line_data = line.strip().split(', ')
            FaceRegion.append([float(line_data[0]), float(line_data[1]), float(line_data[2]), float(line_data[3])])
            continue
                    
        Top = FaceRegion[0][0] - 20.0; FaceRegion_new.append(Top)
        Bottom = FaceRegion[0][1] + 15.0; FaceRegion_new.append(Bottom)
        Left = FaceRegion[0][2] - 6.0; FaceRegion_new.append(Left)
        Right = FaceRegion[0][3] + 6.0; FaceRegion_new.append(Right)
        FaceRegion_new = torch.from_numpy(np.array(FaceRegion_new)).unsqueeze(0)
        
        ##### 3DMM
        tdmm_path = image_path.replace('.jpg', '_mesh.obj')
        
        tdmm_trimesh = trimesh.load(tdmm_path, process=False)
        tdmm_vertex = torch.from_numpy(tdmm_trimesh.vertices)
        tdmm_vertex_color = torch.from_numpy(tdmm_trimesh.visual.vertex_colors)[:, :3]
        tdmm_face = torch.from_numpy(tdmm_trimesh.faces)
        tdmm_vertex = torch.cat([tdmm_vertex, tdmm_vertex_color], 1).float()
        
        return {
            'name': img_name,
            'img_512': image.unsqueeze(0),
            'img_1024': image_1024.unsqueeze(0),
            'calib_512': calib.unsqueeze(0),
            'calib_1024': calib_1024.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
            'face_region': FaceRegion_new,
            'face_tdmm': tdmm_vertex.unsqueeze(0),
            'tdmm_faces': tdmm_face.unsqueeze(0)
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image_512'], ['image_1024'], ['calib_512'], ['calib_1024'], ['b_min'], ['b_max'], ['face_region'], ['face_tdmm'], ['tdmm_faces'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
                self.netC_coarse.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            gen_mesh_jiff(opt, self.netG, self.netC, self.netC_coarse, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]
    test_regions = [f[:-4] + '_region.txt' for f in test_images]
    test_tdmm = [f[:-4] + '_mesh.obj' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path, region_path, tdmm_path in tqdm.tqdm(zip(test_images, test_masks, test_regions, test_tdmm)):
        try:
            print(image_path, 'subject being processed')
            data = evaluator.load_image(image_path, mask_path, region_path, tdmm_path)
            evaluator.eval(data, True)
        except Exception as e:
           print("error:", e.args)
