import os
import torch
import numpy as np
import argparse
import time
import cv2
import shutil

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#from mtcnn_det import FaceDetect
from detect_utils.detect_head import HeadDetect
from detect_utils.detect_landmarks_in_image import collate_pil
from tdmm_utils.recon_test import recon_test
from detect_utils.preprocess.mtcnn import MTCNN

def Pre_Process_face(out_path, input_dirname, subject_file_name):

    render_path = os.path.join(input_dirname, subject_file_name)
    # prefix
    mask_path = render_path.replace(".jpg", "_mask.png")
    os.makedirs(os.path.join(out_path, 'FACE_CROP', 'FACE_CROP'), exist_ok=True)
    region_path = render_path.replace(".jpg", "_region.txt")
    region_face_path = render_path.replace(".jpg", "_region_face.txt")
    
    # Crop face
    HeadDetect(render_path, region_path, region_face_path)
    
    # detect_landmarks
            
    batchSize = 1
    workers = 0 if os.name == 'nt' else 8
    face_data_dir = os.path.join(input_dirname, "FACE_CROP")
    cropped_face_data = os.path.join(input_dirname, "FACE_LANDMARKS")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
    mtcnn = MTCNN(
        image_size=(128, 128), margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    dataset = datasets.ImageFolder(
        face_data_dir, transform=transforms.Resize((128, 128)))
    dataset.samples = [
        (p, p.replace(face_data_dir, cropped_face_data))
        for p, _ in dataset.samples
    ]
    loader = DataLoader(
        dataset,
        num_workers = workers,
        batch_size = batchSize,
        collate_fn=collate_pil
    )
            
    for i, (x, y) in enumerate(loader):
        x = mtcnn(x, save_path=y, save_landmarks=True)
    
    # reconstruct 3dmm and project to 2d image from Accurate3D
    cropped_face_dir = os.path.join(input_dirname, 'FACE_LANDMARKS', 'FACE_CROP')
    recon_test(cropped_face_dir, input_dirname, region_face_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/userhome/cs/yukang/data/RenderPeople_ex/rp_mei_posed_001_OBJ_512/RENDER', help="path to dataset directory, with prt files")
    args = parser.parse_args()
    
    paths = []
    for root, dirs, files in os.walk(args.input_dir):
        path = root.split(os.sep)
        
        for file in files:
            if file.endswith(".jpg") :
                subject_file = os.path.join(root, file).split("/")[-1]
                if subject_file not in paths:
                    paths.append(subject_file)
    N_subject = len(paths)
    print("# test subject found: ", N_subject)
    paths = sorted(paths)
    
    for i, p in enumerate(paths) :
        print(p, 'subject being processes')
        
        subject_file_name = p

        Pre_Process_face(args.input_dir, args.input_dir, subject_file_name)
        
        shutil.rmtree(os.path.join(args.input_dir, 'FACE_CROP'))
        shutil.rmtree(os.path.join(args.input_dir, 'FACE_LANDMARKS'))
        
