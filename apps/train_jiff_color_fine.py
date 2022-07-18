import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

# get options
opt = BaseOptions().parse()

def train_color(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % int(opt.gpu_ids.split(',')[0]))

    train_dataset = TrainDataset_jiff(opt, phase='train')
    test_dataset = TrainDataset_jiff(opt, phase='test')

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    lr = opt.learning_rate

    # Always use resnet for color regression
    netC = ResBlkPIFuNet_fine(opt).to(device=cuda)
    optimizerC = torch.optim.Adam(netC.parameters(), lr=opt.learning_rate)
    netC_coarse = ResBlkPIFuNet(opt).to(device=cuda)

    def set_train():
        netG.eval()
        netC_coarse.eval()
        netC.train()

    def set_eval():
        netG.eval()
        netC_coarse.eval()
        netC.eval()

    print('Using NetworkG: ', netG.name, 'networkC: ', netC.name)

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))
    else:
        model_path_G = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        print('loading for net G ...', model_path_G)
        netG.load_state_dict(torch.load(model_path_G, map_location=cuda))
    
    if opt.load_netC_checkpoint_path is not None:
        print('loading for net C ...', opt.load_netC_checkpoint_path)
        netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
    
    if opt.load_netC_coarse_checkpoint_path is not None:
        print('loading for net C coarse ...', opt.load_netC_coarse_checkpoint_path)
        netC_coarse.load_state_dict(torch.load(opt.load_netC_coarse_checkpoint_path, map_location=cuda))
        
    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path_C = '%s/%s/netC_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path_C = '%s/%s/netC_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)

        print('Resuming from ', model_path_C)
        netC.load_state_dict(torch.load(model_path_C, map_location=cuda))
    
        
    if len(opt.gpu_ids.split(",")) > 1:
        gpus = [ int(i) for i in opt.gpu_ids.split(",") ]
        print("Train with multiple GPUs: {} GPUs used.".format(len(gpus)))
        netC_dp = torch.nn.DataParallel(netC, device_ids=gpus).to(device=cuda) # 0.0537s

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()
            # retrieve the data
            image_1024_tensor = train_data['img_1024'].to(device=cuda)
            image_512_tensor = train_data['img_512'].to(device=cuda)
            calib_1024_tensor = train_data['calib_1024'].to(device=cuda)
            calib_512_tensor = train_data['calib_512'].to(device=cuda)
            color_sample_tensor = train_data['color_samples'].to(device=cuda)
            tdmm_tensor = train_data['face_tdmm'].to(device=cuda)
            face_region_tensor = train_data['face_region'].to(device=cuda)
            view_id_tensor = train_data['view_id'].to(device=cuda)

            image_1024_tensor, calib_1024_tensor = reshape_multiview_tensors(image_1024_tensor, calib_1024_tensor)
            image_512_tensor, calib_512_tensor = reshape_multiview_tensors(image_512_tensor, calib_512_tensor)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)
            
            
            rgb_tensor = train_data['rgbs'].to(device=cuda)

            with torch.no_grad():
                netG.filter(image_512_tensor)
                netC_coarse.filter(image_512_tensor)
                netC_coarse.attach(netG.get_im_feat())
                netC_coarse.query(points=color_sample_tensor, calibs=calib_512_tensor, labels=rgb_tensor)
                
                            
            if len(opt.gpu_ids.split(",")) > 1:
                resC, error = netC_dp.forward(image_1024_tensor, netC_coarse.get_mlp_feature(), color_sample_tensor, calib_1024_tensor, tdmm_vertex=tdmm_tensor, face_region=face_region_tensor, view_id=view_id_tensor, labels=rgb_tensor)
            else:
                resC, error = netC.forward(image_1024_tensor, netC_coarse.get_mlp_feature(), color_sample_tensor, calib_1024_tensor, tdmm_vertex=tdmm_tensor, face_region=face_region_tensor, view_id=view_id_tensor, labels=rgb_tensor)
                
            if len(opt.gpu_ids.split(",")) > 1 :
                error = error.mean()
                            
            with torch.autograd.set_detect_anomaly(True):
                optimizerC.zero_grad()
                error.backward()
                optimizerC.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | dataT: {6:.05f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader),
                        error.item(),
                        lr,
                        iter_start_time - iter_data_time,
                        iter_net_time - iter_start_time, int(eta // 60),
                        int(eta - 60 * (eta // 60))))

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netC.state_dict(), '%s/%s/netC_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netC.state_dict(), '%s/%s/netC_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            if train_idx % opt.freq_save_ply == 0:
                save_path = '%s/%s/pred_col.ply' % (opt.results_path, opt.name)
                rgb = resC[0].transpose(0, 1).cpu() * 0.5 + 0.5
                points = color_sample_tensor[0].transpose(0, 1).cpu()
                save_samples_rgb(save_path, points.detach().numpy(), rgb.detach().numpy())

            iter_data_time = time.time()

        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_num_eval:
                test_losses = {}
                print('calc error (test) ...')
                if len(opt.gpu_ids.split(",")) > 1:
                    test_color_error = calc_error_color(opt, netG, netC_dp, netC_coarse, cuda, test_dataset, 100)
                else:
                    test_color_error = calc_error_color(opt, netG, netC, netC_coarse, cuda, test_dataset, 100)
                print('eval test | color error:', test_color_error)
                test_losses['test_color'] = test_color_error

                print('calc error (train) ...')
                train_dataset.is_train = False
                train_color_error = calc_error_color(opt, netG, netC, netC_coarse, cuda, train_dataset, 100)
                train_dataset.is_train = True
                print('eval train | color error:', train_color_error)
                test_losses['train_color'] = train_color_error


if __name__ == '__main__':
    train_color(opt)
