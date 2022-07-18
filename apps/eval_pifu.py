import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
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
        netG = HGPIFuNet_PIFu(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))
                
        netC = ResBlkPIFuNet(opt).to(device=cuda)
        print('Using Network: ', netC.name)
        
        netC.load_state_dict(torch.load(opt.load_netC_coarse_checkpoint_path, map_location=cuda))
        
        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path):
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
        
        ##### Mask
        mask = Image.open(mask_path).convert('L')
        
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        
        ##### image
        image = Image.open(image_path).convert('RGB')
        
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'], ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            
            gen_mesh_pifu(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)
    
    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        try:
            print(image_path, 'subject being processed')
            data = evaluator.load_image(image_path, mask_path)
            evaluator.eval(data, True)
        except Exception as e:
           print("error:", e.args)
