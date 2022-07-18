# code modified from https://github.com/jiuxianghedonglu/AnimeHeadDetection/blob/master/detect_image.py
import numpy as np
import os
import sys
import face_recognition
import torch
from PIL import Image
from .predictor import Predictor
from .mesh import load_obj_mesh
from .preprocess.mtcnn import MTCNN

def HeadDetect(image_path, region_path, region_face_path):
    # Detect the head region for inference.
    # Instead of only face detection, the head detection may make the
    # reconstruction more smooth around the bounary between face and body.
    
    # We may still need to offset the bbox a little bit to
    # take the misdetection into account
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = Predictor(device=device)
    
    # head detection
    img = predictor.read_img(image_path)
    x = predictor.process_img(img)
    predictions = predictor.predict(x)
    bbox = predictions['boxes'][0]
    
    # face detection for rescaling 3dmm
    image = face_recognition.load_image_file(image_path)
    mtcnn_f = MTCNN(keep_all=True)
    face_det, probs= mtcnn_f.detect(image)
    bbox_face = face_det[0]
    # bbox and offset
    
    # this bbox may just contain the face region, instead of the full head,
    # Therefore, we offset the bbox to make sure it include the full-head region.
    # for large-pose case, you may modify the offset correspondingly
    top = bbox[1] - 10.0
    bottom = bbox[3] + 10.0
    left = bbox[0] - 4.0
    right = bbox[2] + 4.0
    
    # locate z region for transforming 3dmm
    subject = image_path.split('/')[-1]
    obj_subject = 'result_' + subject
    obj_subject = obj_subject.replace('jpg', 'obj')
    obj_path = os.path.join('./results/coarse_recontruction', obj_subject)
    
    obj_tuple = load_obj_mesh(obj_path)
    vertices = obj_tuple[0]
    ymin = (1.0 - bbox_face[3] / 256.0) * 90 + 100
    ymax = (1.0 - bbox_face[1] / 256.0) * 90 + 100
    xmin = bbox_face[0] / 2.0 - 128
    xmax = bbox_face[2] / 2.0 - 128
            
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    in_face = (y[:] >= ymin) & (y[:] <= (ymax)) & (x[:] >= xmin) & (x[:] <= xmax)
    vertices_face = vertices[in_face, :]
    
    z_face = vertices_face[:, 2]
    zmax = np.max(z_face)
    zmin = np.min(z_face)
    
    # save region path
    with open(region_path, 'a') as f:
        f.write("{}, {}, {}, {}\n".format(top, bottom, left, right))
    with open(region_face_path, 'a') as f:
        f.write("{}, {}, {}, {}, {}, {}\n".format(bbox_face[1], bbox_face[3], bbox_face[0], bbox_face[2], zmin, zmax))
        
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
    
    image = face_recognition.load_image_file(image_path)
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    
    outFace_img_path = os.path.join(os.path.dirname(image_path), 'FACE_CROP', 'FACE_CROP', subject)
    pil_image.save(outFace_img_path)
