import numpy as np
import os
import torch
import trimesh
import argparse
import math
from tqdm import tqdm

from detect_utils.mesh import load_obj_mesh, make_rotate, save_obj_mesh_3dmm

if __name__ == '__main__':
    '''
    Align the front-view 3dmm mesh with ground-truth mesh
    rotate the front-view 3dmm mesh to certain angles
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/userhome/cs/yukang/data/RenderPeople_ex/rp_mei_posed_001_OBJ_512/RENDER', help="path to dataset directory, with prt files")
    args = parser.parse_args()
    
    gt_path = os.path.join(args.input_dir, 'GEO/OBJ')
    tdmm_path = os.path.join(args.input_dir, 'FACE_3DMM')
    for obj in os.listdir(gt_path):
    
        obj_path = os.path.join(gt_path, obj, obj+'.obj')
        subject = obj
        
        tdmm_subject_path = os.path.join(tdmm_path, subject)
        
        tdmm_mesh_path = os.path.join(tdmm_path, subject, '0_0_00_mesh.obj')
                
        pifu_obj = pifu_obj = trimesh.load(obj_path, process=False)
        pifu_vs = pifu_obj.vertices
                    
        tdmm_tuple = load_obj_mesh(tdmm_mesh_path, with_normal=True, with_texture=True, with_vertex_color=True)
                
        tdmm_vs = tdmm_tuple[0]
        tdmm_face = tdmm_tuple[1]
        tdmm_norm = tdmm_tuple[2]
        tdmm_vertex_color = tdmm_tuple[3]
        tdmm_face_normal = tdmm_tuple[4]
                
        _, icp_tdmm_vs, _ = trimesh.registration.icp(tdmm_vs, pifu_vs)
        
        save_obj_mesh_3dmm(tdmm_mesh_path, icp_tdmm_vs, tdmm_vertex_color, tdmm_face)
        
        for angle in tqdm(range(0, 360, 1)):
        
                R = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(angle), 0))
                rotate_vertices = np.transpose(icp_tdmm_vs)
                rotate_vertices = np.transpose(np.dot(R, rotate_vertices))
                rotated_tdmm_path = os.path.join(tdmm_subject_path, str(angle) + '_0_00_mesh.obj')
                
                save_obj_mesh_3dmm(rotated_tdmm_path, rotate_vertices, tdmm_vertex_color, tdmm_face)
                
