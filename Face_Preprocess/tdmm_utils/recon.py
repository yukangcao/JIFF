import os
import glob
import torch
import numpy as np
import math

from .models.resnet_50 import resnet50_use
from .load_data import transfer_BFM09, BFM, load_img, Preprocess, save_obj
from .reconstruction_mesh import reconstruction, render_img, transform_face_shape, estimate_intrinsic

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


def recon(faceDetect_path, faceProject_path, region_path):
    ## input and output folder
    image_path = faceDetect_path
    save_path = faceProject_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    img_list = glob.glob(image_path + '/**/' + '*.png', recursive=True)
    img_list += glob.glob(image_path + '/**/' + '*.jpg', recursive=True)

    ## read BFM face model
    ## transfer original BFM model to our model
    if not os.path.isfile('./Face_Preprocess/BFM/BFM_model_front.mat'):
        transfer_BFM09()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    bfm = BFM(r'./Face_Preprocess/BFM/BFM_model_front.mat', device)

    ## read standard landmarks for preprocessing images
    lm3D = bfm.load_lm3d()

    model = resnet50_use().to(device)
    model.load_state_dict(torch.load(r'./Face_Preprocess/tdmm_utils/models/params.pt'))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    for file in img_list:
        ## load images and corresponding 5 facial landmarks
                
        theta_value = file.split('/')[-1].split('_')[0]
        theta = int(theta_value)
        
        theta_ori = file.split('/')[-1]
        file = file.replace(theta_ori, '0_0_00.jpg')
        img, lm = load_img(file, file.replace('jpg', 'txt'))

        # preprocess input image
        input_img_org, lm_new, transform_params = Preprocess(img, lm, lm3D)

        input_img = input_img_org.astype(np.float32)
        input_img = torch.from_numpy(input_img).permute(0, 3, 1, 2)
        # the input_img is BGR
        input_img = input_img.to(device)

        arr_coef = model(input_img)

        coef = torch.cat(arr_coef, 1)

        ## reconstruct 3D face with output coefficients and face model
        face_shape, face_texture, face_color, landmarks_2d, z_buffer, angles, translation, gamma = reconstruction(coef, bfm)

        fx, px, fy, py = estimate_intrinsic(landmarks_2d, transform_params, z_buffer, face_shape, bfm, angles, translation)
        
        face_region_path = os.path.join(region_path, file.split('/')[-1]).replace('jpg', 'txt')
        
        fp = open(face_region_path, 'r')
        lines = fp.readlines()
        fp.close()
        FaceRegion = []
        for line in lines:
            line_data = line.strip().split(', ')
            FaceRegion.append([float(line_data[0]), float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]), float(line_data[5])])
            continue
            
        xmin = FaceRegion[0][2]
        xmax = FaceRegion[0][3]
        ymin = FaceRegion[0][0]
        ymax = FaceRegion[0][1]
        
        zmin = FaceRegion[0][4]
        zmax = FaceRegion[0][5]
        
        x_max = FaceRegion[0][3] / 2.0 - 128.0
        x_min = FaceRegion[0][2] / 2.0 - 128.0
        
        y_min = (1.0 - FaceRegion[0][1] / 256.0) * 90 + 100
        y_max = (1.0 - FaceRegion[0][0] / 256.0) * 90 + 100
        
        ysize = ymax - ymin
        
        face_shape_t = transform_face_shape(face_shape, angles, translation)
        face_color = face_color / 255.0
        face_shape_t[:, :, 2] = 10.0 - face_shape_t[:, :, 2]
        
        face_shape = face_shape.detach().cpu().numpy()
        face_color = face_color.detach().cpu().numpy()
        
        face_shape = np.squeeze(face_shape)
        
        face_shape_t = face_shape_t.detach().cpu().numpy()
        face_shape_t = np.squeeze(face_shape_t)

        ## rescale the 3DMM mesh and roughly align with the ground-truth meshes, according to the cropped image's size
        face_shape_t[:, 0] = face_shape_t[:, 0] - np.mean(face_shape_t[:, 0])
        face_shape_t[:, 1] = face_shape_t[:, 1] - np.mean(face_shape_t[:, 1])
        face_shape_t[:, 2] = face_shape_t[:, 2] - np.mean(face_shape_t[:, 2])

        face_shape_t[:, 1] = face_shape_t[:, 1] - np.min(face_shape_t[:, 1])
        face_shape_t[:, 2] = face_shape_t[:, 2] * -1.0

        vmin = face_shape_t.min(0)
        vmax = face_shape_t.max(0)
        
        up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
        
        vmed = np.median(face_shape_t, 0)
        
        vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
        
        height = ysize / (512.0 * 13 / 15) * 180.0

        a = vmax[up_axis] - vmin[up_axis]
        if a < 1.5:
            a = 1.5
        y_scale = height / a

        N = np.eye(4)
        N[:3, :3] = y_scale*np.eye(3)
        N[:3, 3] = -y_scale*vmed

        ones = np.ones(face_shape_t.shape[0])
        ones = ones[:, np.newaxis]
        vertices_f = np.concatenate((face_shape_t, ones), axis=1)
        vertices_s = np.dot(vertices_f, N)
        face_shape_t = vertices_s[:, :3]


        z_vertices = face_shape_t[:, 2]
        z_vertices = z_vertices[np.newaxis, :]
        z_vertices_max = np.max(z_vertices)
        z_add = zmax - z_vertices_max

        x_vertices = face_shape_t[:, 0]
        x_vertices = x_vertices[np.newaxis, :]
        x_vertices_max = np.max(x_vertices)
        x_vertices_min = np.min(x_vertices)
        x_add = ((x_max - x_vertices_max) + (x_min - x_vertices_min)) / 2.0

        y_vertices = face_shape_t[:, 1]
        y_vertices = y_vertices[np.newaxis, :]
        y_vertices_max = np.max(y_vertices)
        y_vertices_min = np.min(y_vertices)
        
        y_add = ((y_max - y_vertices_max) + (y_min - y_vertices_min)) / 2.0

        face_shape_t[:, 1] = face_shape_t[:, 1] + y_add
        face_shape_t[:, 2] = face_shape_t[:, 2] + z_add
        face_shape_t[:, 0] = face_shape_t[:, 0] + x_add


        Rotation = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(theta), 0))
        face_shape_t = np.transpose(face_shape_t)
        face_shape_t = np.transpose(np.dot(Rotation, face_shape_t))
        
        face_color = np.squeeze(face_color)
        save_obj(os.path.join(save_path, file.replace('0_0_00.jpg', theta_ori).split("/")[-1]).replace('.jpg', '_mesh.obj'), face_shape_t, bfm.tri, np.clip(face_color, 0, 1.0))  # 3D reconstruction face (in posed camera view)


if __name__ == '__main__':
    recon()
