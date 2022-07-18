import numpy as np
import os
import torch
import trimesh
import argparse
import math

from detect_utils.mesh import load_obj_mesh, make_rotate, save_obj_mesh_3dmm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pifu', '--PIFu_path', type=str, default='', help="path to PIFu reconstruction")
        
    parser.add_argument('-test', '--test_path', type=str, default='', help="path to 3dmm mesh")
    args = parser.parse_args()

    
    for file in os.listdir(args.test_path):
        if file.endswith('jpg'):
            subject = file.split('.')[0]
            pifu_path = os.path.join(args.PIFu_path, 'result_'+subject+'.obj')
            tdmm_path = os.path.join(args.test_path, subject+'_mesh.obj')
            
            pifu_obj = pifu_obj = trimesh.load(pifu_path, process=False)
            pifu_vs = pifu_obj.vertices
                        
            tdmm_tuple = load_obj_mesh(tdmm_path, with_normal=True, with_texture=True, with_vertex_color=True)
                    
            tdmm_vs = tdmm_tuple[0]
            tdmm_face = tdmm_tuple[1]
            tdmm_norm = tdmm_tuple[2]
            tdmm_vertex_color = tdmm_tuple[3]
            tdmm_face_normal = tdmm_tuple[4]
                    
            _, icp_tdmm_vs, _ = trimesh.registration.icp(tdmm_vs, pifu_vs)
                    
            rotated_tdmm_path = tdmm_path
                    
            save_obj_mesh_3dmm(rotated_tdmm_path, icp_tdmm_vs, tdmm_vertex_color, tdmm_face)
            
