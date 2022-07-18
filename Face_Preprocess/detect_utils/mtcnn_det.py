import os.path as osp
import os

import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from .utils import utils
import dlib
import face_recognition
from PIL import Image, ImageDraw, ImageFile
from .preprocess.mtcnn import MTCNN
from .mesh import load_obj_mesh

def FaceDetect(img_path, mask_path, region_path, crop_path):
    # detect the face region to be 128x128
    # save the face image and mask for 3dmm estimation, and the face bbox with offset
    # output the Face and Face_mask image for generating 3dmm
    # face_boxes = FaceBoxes(timer_flag=True)

    img_ori = img_path
    
    image = face_recognition.load_image_file(img_ori)
    
    mtcnn_f = MTCNN(keep_all=True)
    face_det, probs= mtcnn_f.detect(image)
    if face_det is None:
        return None, None
    else:
        top, bottom, right, left = float(face_det[0][1]), float(face_det[0][3]), float(face_det[0][2]), float(face_det[0][0])
        
        if img_ori.split('/')[-1] == '0_0_00.jpg':
            print('0_0_00.txt')
            print(img_ori)

            # find and record z value for the face region.
            imgOri = img_ori.split('/')
            obj_path = img_ori.replace('RENDER', 'GEO/OBJ')
            obj_path = os.path.join(os.path.dirname(obj_path), imgOri[-2] + '.obj')

            obj_tuple = load_obj_mesh(obj_path)
            vertices = obj_tuple[0]

            ymin = (1.0 - bottom / 256.0) * 90 + 100
            ymax = (1.0 - top / 256.0) * 90 + 100
            xmin = left / 2.0 - 128
            xmax = right / 2.0 - 128
            
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            
            in_face = (y[:] >= ymin) & (y[:] <= (ymax)) & (x[:] >= xmin) & (x[:] <= xmax)
            vertices_face = vertices[in_face, :]

            z_face = vertices_face[:, 2]
            zmax = np.max(z_face)
            zmin = np.min(z_face)
            
            with open(region_path, 'a') as f:
                f.write("{}, {}, {}, {}, {}, {}\n".format(top, bottom, left, right, zmin, zmax))
        else:
            with open(region_path, 'a') as f:
                f.write("{}, {}, {}, {}\n".format(top, bottom, left, right))
            
        height = bottom - top
        Tchange = int((128 - height) / 2)
        Bchange = 128 - height - Tchange
    
        width = right - left
        Rchange = int((128 - width) / 2)
        Lchange = 128 - width - Rchange
    
        top = int(top - Tchange)
        right = int(right + Rchange)
        bottom = int(bottom + Bchange)
        left = int(left - Lchange)
        if top < 0:
            bottom = bottom - top
            top = 0
        if left < 0:
            right = right - left
            left = 0
    
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
    # img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (512, 512))
    
        outFace_img_path = img_ori.replace('RENDER', 'FACE_CROP')
        pil_image.save(crop_path)
    # img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    
        mask_ori = mask_path
        mask = face_recognition.load_image_file(mask_ori)
        mask_image = mask[top:bottom, left:right]
        pil_mask = Image.fromarray(mask_image)
        
        return pil_image, pil_mask
