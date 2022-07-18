#### normal related function borrowed from PIFu (https://github.com/shunsukesaito/PIFu)

from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import time

import argparse
import glob


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line:
            split_line = line.strip().split(' ', 1)
            if len(split_line) < 2:
                continue

            prefix, data = split_line[0], split_line[1]
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if data:
                    split_data = data.strip().split(' ')

                    # assume texture maps are in the same level
                    # WARNING: do not include space in your filename!!
                    if 'map' in prefix:
                        material[prefix] = split_data[-1].split('\\')[-1]
                    elif len(split_data) > 1:
                        material[prefix] = tuple(float(d) for d in split_data)
                    else:
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)

    return materials


def load_obj_mesh_mtl(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    # face per material
    face_data_mat = {}
    face_norm_data_mat = {}
    face_uv_data_mat = {}

    # current material name
    mtl_data = None
    cur_mat = None

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
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'mtllib':
            mtl_data = read_mtlfile(mesh_file.replace(mesh_file.split('/')[-1],values[1]))
        elif values[0] == 'usemtl':
            cur_mat = values[1]
        elif values[0] == 'f':
            # local triangle data
            l_face_data = []
            l_face_uv_data = []
            l_face_norm_data = []

            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, values[1:4]))
                l_face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, [values[3], values[4], values[1]]))
                l_face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, values[1:4]))
                l_face_data.append(f) # !!
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, values[1:4]))
                    l_face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, [values[3], values[4], values[1]]))
                    l_face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, values[1:4]))
                    l_face_uv_data.append(f) # !!
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, values[1:4]))
                    l_face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, [values[3], values[4], values[1]]))
                    l_face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, values[1:4]))
                    l_face_norm_data.append(f)
            
            face_data += l_face_data # !!
            face_uv_data += l_face_uv_data # !!
            face_norm_data += l_face_norm_data

            if cur_mat is not None:
                if cur_mat not in face_data_mat.keys():
                    face_data_mat[cur_mat] = []
                if cur_mat not in face_uv_data_mat.keys():
                    face_uv_data_mat[cur_mat] = []
                if cur_mat not in face_norm_data_mat.keys():
                    face_norm_data_mat[cur_mat] = []
                face_data_mat[cur_mat] += l_face_data
                face_uv_data_mat[cur_mat] += l_face_uv_data
                face_norm_data_mat[cur_mat] += l_face_norm_data

    vertices = np.array(vertex_data)
    faces = np.array(face_data)

    norms = compute_normal(vertices, faces)
    norms = normalize_v3(norms)
    face_normals = np.array(face_norm_data)

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data)

    out_tuple = (vertices, faces, norms, face_normals, uvs, face_uvs)

    if cur_mat is not None and mtl_data is not None:
        for key in face_data_mat:
            face_data_mat[key] = np.array(face_data_mat[key])
            face_uv_data_mat[key] = np.array(face_uv_data_mat[key])
            face_norm_data_mat[key] = np.array(face_norm_data_mat[key])
        
        out_tuple += (face_data_mat, face_norm_data_mat, face_uv_data_mat, mtl_data)

    return out_tuple
    
def save_obj_mesh(mesh_path, verts, uvs, faces, face_uvs):
    file = open(mesh_path, 'w')
    file.write('mtllib material0.mtl\nusemtl material0\n')
    for v in verts:
        file.write('v %.8f %.8f %.8f\n' % (v[0], v[1], v[2]))
#    for f in faces:
#        f_plus = f + 1
#        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    for uv in uvs:
        file.write('vt %.8f %.8f\n' % (uv[0], uv[1]))
    for f, f_uv in zip(faces, face_uvs):
        f_plus = f + 1
        f_uv_plus = f_uv + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_uv_plus[0], f_plus[1], f_uv_plus[1], f_plus[2], f_uv_plus[2]))
    file.close()
    

if __name__ == '__main__':
    ##### Input Arguments #####
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/disk2/ykcao/THuman2.0/THuman2.0_raw', help="Input data directory.")
    parser.add_argument('-o', '--output_dir', type=str,  help='Output directory. If not provided, input files will be overwritten.')
    parser.add_argument('-j', type=int, default=10, help='number of workers')

    args = parser.parse_args()
    
    # Normalize data paths
    args.input_dir = os.path.normpath(args.input_dir)
    if args.output_dir :
        args.output_dir = os.path.normpath(args.output_dir)
    print("Input directory: ", args.input_dir)
    print("Output directory: ", args.output_dir)

    ##### Get Input Paths #####
    # Find paths of meshes to process
    input_paths = glob.glob(os.path.join(args.input_dir, "**", "**" + ".obj"), recursive=True)
    num_paths = len(input_paths)
    print("# .obj meshes found: ", num_paths)
    
    # Set output paths
    if args.output_dir :
        os.makedirs(args.output_dir, exist_ok=True)
        output_paths = [ p.replace(args.input_dir, args.output_dir) for p in input_paths ]
    else :
        # If output_dir is not provided, write files at the input locations
        output_paths = input_paths
        
    input_paths = sorted(input_paths)
    
    #### Process Mesh to height 180#####
    for mesh_file in input_paths:
        
        print(mesh_file, 'being processed')
        out_tuple = load_obj_mesh_mtl(mesh_file)

        vertices = out_tuple[0]
        faces = out_tuple[1]
        norms = out_tuple[2]
        uvs = out_tuple[4]
        face_uvs = out_tuple[5]

        vertices[:, 0] = vertices[:, 0] - np.mean(vertices[:, 0])
        vertices[:, 1] = vertices[:, 1] - np.mean(vertices[:, 1])
        vertices[:, 2] = vertices[:, 2] - np.mean(vertices[:, 2])
        vertices[:, 1] = vertices[:, 1] - np.min(vertices[:, 1])

        vmin = vertices.min(0)
        vmax = vertices.max(0)
        
        up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
        vmed = np.median(vertices, 0)
        vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
        y_scale = 180.0 / (vmax[up_axis] - vmin[up_axis])

        N = np.eye(4)
        N[:3, :3] = y_scale*np.eye(3)
        N[:3, 3] = -y_scale*vmed
        
        ones = np.ones(vertices.shape[0])
        ones = ones[:, np.newaxis]
        vertices_f = np.concatenate((vertices, ones), axis=1)
        
        vertices_s = np.dot(vertices_f, N)
        vertices = vertices_s[:, :3]
        
        save_obj_mesh(mesh_file, vertices, uvs, faces, face_uvs)
