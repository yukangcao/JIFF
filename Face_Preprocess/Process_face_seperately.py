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

from detect_utils.mtcnn_det import FaceDetect
from detect_utils.detect_landmarks_in_image import collate_pil
from tdmm_utils.recon import recon
from detect_utils.preprocess.mtcnn import MTCNN

def Pre_Process_face(out_path, input_dirname, subject_file_name):
    '''
    steps:
    1. crop and locate face region
    2. detect landmarks for 3dmm estimation
    3. estimate 3dmm mesh
    '''
    subject_dir = os.path.join(input_dirname, subject_file_name)
    # prefix
    for JPG in os.listdir(subject_dir):
        render_path = os.path.join(subject_dir, JPG)
        mask_path = os.path.join(os.path.dirname(input_dirname), "MASK", subject_file_name, JPG).replace("jpg", "png")
        os.makedirs(os.path.join(out_path, 'FACE_REGION', subject_file_name, subject_file_name), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'FACE_CROP', subject_file_name, subject_file_name), exist_ok=True)
        region_path = os.path.join(os.path.dirname(input_dirname), "FACE_REGION", subject_file_name, subject_file_name, JPG).replace("jpg", "txt")
        crop_path = os.path.join(os.path.dirname(input_dirname), "FACE_CROP", subject_file_name, subject_file_name, JPG)
    
    # Crop face
        face_image, face_mask_image = FaceDetect(render_path, mask_path, region_path, crop_path)
    print("# Done Cropping face: ", subject_dir)
    
    # detect_landmarks
            
    batchSize = 1
    workers = 0 if os.name == 'nt' else 8
    face_data_dir = os.path.join(os.path.dirname(input_dirname), "FACE_CROP", subject_file_name)
    cropped_face_data = os.path.join(os.path.dirname(input_dirname), "FACE_LANDMARKS", subject_file_name)
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
    print("# Done Landmarks Detecting: ", subject_dir)
    
    # reconstruct 3dmm and project to 2d image from Accurate3D
    faceProjected_dir = os.path.join(os.path.dirname(input_dirname), "FACE_3DMM", subject_file_name)
    cropped_face_dir = os.path.join(os.path.dirname(input_dirname), "FACE_LANDMARKS", subject_file_name, subject_file_name)
    faceregion_path = os.path.join(os.path.dirname(input_dirname), "FACE_REGION", subject_file_name, subject_file_name)
    recon(cropped_face_dir, faceProjected_dir, faceregion_path)
    print("# Done 3DMM estimating: ", subject_dir)

    

if __name__ == '__main__':
    shs = np.load('./env_sh.npy')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/userhome/cs/yukang/data/RenderPeople_ex/rp_mei_posed_001_OBJ_512/RENDER', help="path to dataset directory, with prt files")
    parser.add_argument('-o', '--output_dir', type=str, default='/userhome/cs/yukang/data/RenderPeople_ex/rp_mei_posed_001_OBJ_512', help='output directory')
    parser.add_argument('-l', '--log_path',  type=str, default="./log/log.txt", help='log file path')
    args = parser.parse_args()
    
    paths = []
    for root, dirs, files in os.walk(args.input_dir):
        path = root.split(os.sep)
        
        for file in files:
            if file.endswith(".jpg") :
                subject_file = os.path.join(root, file).split("/")[-2]
                if subject_file not in paths:
                    paths.append(subject_file)
    N_subject = len(paths)
    print("# subject found: ", N_subject)
    paths = sorted(paths)
    processed_files = []
    if (os.path.isfile(args.log_path)) :
        with open(args.log_path, 'r') as f :
            processed_files = [ line.strip() for line in f ]
    else :
        # Prepare log file
        log_dir = os.path.dirname(args.log_path)
        os.makedirs(log_dir, exist_ok=True)
        
    print("# processed files: ", len(processed_files))
    
    for i, p in enumerate(paths) :
        print(p, 'subject being processes')
        # Skip file if it is processed alraedy
        os.makedirs(os.path.join(args.output_dir, 'FACE_CROP', p), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'FACE_LANDMARKS', p), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'FACE_3DMM', p), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'FACE_REGION', p), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'FILE_NUMBER', p), exist_ok=True)
        
        if p in processed_files :
            print("File processed alraedy: ", p)
            continue
        
        subject_file_name = p

        
        Pre_Process_face(args.output_dir, args.input_dir, subject_file_name)
        
        shutil.rmtree(os.path.join(args.output_dir, 'FACE_CROP'))
        shutil.rmtree(os.path.join(args.output_dir, 'FACE_LANDMARKS'))
        
        # Mark the current file as processed
        with open(args.log_path, 'a') as f :
            f.write("{}\n".format(p))
    ## change the directory for FACE_REGION
    region_dir = os.path.join(args.output_dir, 'FACE_REGION')
    paths = []
    for file in os.listdir(region_dir):
        if file.startswith('0') and file not in paths:
            paths.append(file)
            file_dir = os.path.join(region_dir, file, file) # /results_xx_xxx/rsulst_xx_xxx/
            if os.path.exists(file_dir):
                for files in os.listdir(file_dir): # jpg
                        dst = os.path.join(region_dir, file, files)
                        files_path = os.path.join(file_dir, files)
                        shutil.move(files_path, dst)
                shutil.rmtree(file_dir)
                
    number_dir = os.path.join(args.output_dir, 'FILE_NUMBER')
    for subject in os.listdir(region_dir):
        subject_dir = os.path.join(region_dir, subject)
        number_file = os.path.join(number_dir, subject)
        os.makedirs(number_file, exist_ok=True)
        file_name = os.path.join(number_file, 'number.txt')
        number = []
        for view in os.listdir(subject_dir):
            view = int(view.split('_')[0])
            with open(file_name, 'a') as f:
                f.write("{}\n". format(view))
