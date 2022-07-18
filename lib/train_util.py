import torch
import numpy as np
import trimesh
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm
#from .model.MeshNormalizer import MeshNormalizer


def MeshNormalizer(tdmm_vertices):
    py_min = -28
    py_max = 228
    tdmm_vertices_norm = tdmm_vertices
    tdmm_vertices_norm = tdmm_vertices_norm.permute(0, 2, 1)

    tdmm_vertices_norm[:, 1, :] = tdmm_vertices_norm[:, 1, :] - py_min
    tdmm_vertices_norm = tdmm_vertices_norm / (py_max - py_min)
    tdmm_vertices_norm[:, 1, :] = tdmm_vertices_norm[:, 1, :] - 0.5
        
    return py_min, py_max, tdmm_vertices_norm
    
def normalize_3d_coordinate(p, padding=0.1):
    '''
    Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor
    
def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh_pifu(opt, netG, netC, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    
    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())
    
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)
            
        # get PIFu's color
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T
            
        save_obj_mesh_with_color(save_path, verts, faces, color)
        
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def gen_mesh_jiff(opt, netG, netC, netC_coarse, cuda, data, save_path, use_octree=True):

    image_tensor = data['img_1024'].to(device=cuda)
    calib_tensor = data['calib_1024'].to(device=cuda)
    image_512_tensor = data['img_512'].to(device=cuda)
    calib_512_tensor = data['calib_512'].to(device=cuda)
    tdmm_tensor = data['face_tdmm'].to(device=cuda)
    face_region_tensor = data['face_region'].to(device=cuda)
        
    tdmm_face_tensor = data['tdmm_faces'].to(device=cuda)
    
    tdmm_vertex = tdmm_tensor[:, :, :3].clone()
    tdmm_color = tdmm_tensor[:, :, 3:].squeeze(0)
    
    netG.filter(image_512_tensor)
    netC_coarse.filter(image_512_tensor)
    netC_coarse.attach(netG.get_im_feat())
    netC.filter(image_tensor)
    
    py_min, py_max, tdmm_vertices_norm = MeshNormalizer(tdmm_vertex)
    tdmm_vertex_norm = normalize_3d_coordinate(tdmm_vertices_norm.squeeze(0).permute(1, 0), padding=0.1)
    tdmm_vertex_G = tdmm_vertex_norm.unsqueeze(0).float()
    
    netG.filter_tdmm(tdmm_vertex_G)
    tdmm_tensor_norm = torch.cat([tdmm_vertex_norm, tdmm_color], 1).unsqueeze(0).float()
    
    netC.filter_tdmm(tdmm_tensor_norm)
    
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_512_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
        
        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_512_tensor, opt.resolution, b_min, b_max, face_region_tensor, tdmm_tensor, tdmm_face_tensor, py_min, py_max, use_octree=use_octree)
        # getting the verts for color query
        color_verts = verts / 2.0
        color_verts[:, 1] = color_verts[:, 1] + 45.0
        color_verts_tensor = torch.from_numpy(color_verts.T).unsqueeze(0).to(device=cuda).float()
        
        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC_coarse.query(verts_tensor[:, :, left:right], calib_512_tensor)
            calib_tensor = calib_512_tensor * 2.0
            coarse_mlp_features = netC_coarse.get_mlp_feature()
            netC.query(color_verts_tensor[:, :, left:right], calib_tensor, coarse_mlp_features, face_region=face_region_tensor, py_min=py_min, py_max=py_max)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            
            tdmm_tensor = data['face_tdmm'].to(device=cuda).unsqueeze(0)
            face_region_tensor = data['face_region'].to(device=cuda).unsqueeze(0)
            view_id_tensor = data['view_id'].to(device=cuda)
            
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, tdmm_vertex=tdmm_tensor, face_region=face_region_tensor, labels=label_tensor, view_id=view_id_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_error_color(opt, netG, netC, netC_coarse, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_1024_tensor = data['img_1024'].to(device=cuda)
            image_512_tensor = data['img_512'].to(device=cuda)
            calib_1024_tensor = data['calib_1024'].to(device=cuda)
            calib_512_tensor = data['calib_512'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)
                        
            tdmm_tensor = data['face_tdmm'].to(device=cuda).unsqueeze(0)
            face_region_tensor = data['face_region'].to(device=cuda).unsqueeze(0)
            view_id_tensor = data['view_id'].to(device=cuda)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)
        
            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)
            
            netG.filter(image_512_tensor)
            netC_coarse.filter(image_512_tensor)
            netC_coarse.attach(netG.get_im_feat())
            netC_coarse.query(points = color_sample_tensor, calibs=calib_512_tensor, labels=rgb_tensor)
            
            _, errorC = netC.forward(image_1024_tensor, netC_coarse.get_mlp_feature(), color_sample_tensor, calib_1024_tensor, tdmm_vertex=tdmm_tensor, face_region=face_region_tensor, view_id=view_id_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)
    
def calc_error_color_coarse(opt, netG, netC_coarse, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC_coarse.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)
