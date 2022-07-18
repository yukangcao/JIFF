from skimage import measure
import numpy as np
import torch
import trimesh
from .sdf import create_grid, eval_grid_octree, eval_grid
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from pytorch3d.structures import Meshes

def build_triangles(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device=vertices.device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def cal_sdf(verts, faces, points):
    # functions modified from ICON
    
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    
    triangles = build_triangles(verts, faces)
    
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1)

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max, face_region_tensor=None, tdmm_tensor=None, tdmm_face_tensor=None, py_min=None, py_max=None,
                   use_octree=False, num_samples=10000, transform=None): # return verts, faces, normals, values
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.  
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param face_region_tensor: face region bbox
    :param tdmm_tensor: providing tdmm_vertex for sdf calculation
    :param tdmm_face_tensor: providing tdmm_faces for sdf calculation
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max,
                              transform=transform)
                              
    # Then we define the lambda function for cell evaluation
    def eval_func(points, py_min, py_max):
        points = np.expand_dims(points, axis=0) # 在axis=0的位置加一个维度
        points = np.repeat(points, net.num_views, axis=0)  # 延y轴复制num_view次数
        samples = torch.from_numpy(points).to(device=cuda).float() # 转化至tensor，flaot数
        if tdmm_tensor != None:
            tdmm_vertices = tdmm_tensor[:, :, :3]
            
            samples_tdmm = samples.permute(0, 2, 1).contiguous()
            sdf_tdmm = cal_sdf(tdmm_vertices, tdmm_face_tensor, samples_tdmm)
        else:
            sdf_tdmm=None
        
        net.query(samples, calib_tensor, transforms=transform, face_region=face_region_tensor, py_min=py_min, py_max=py_max, sdf=sdf_tdmm)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy() # detach阻断反向传播，返回tensor；cup（）把变量放置cpu上，返回tensor；numpy（）转换为numpy

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, py_min, py_max, num_samples=num_samples) # 运用octree时的eval_func的运用
    else:
        sdf = eval_grid(coords, eval_func, py_min, py_max, num_samples=num_samples) # 不运用octree时的eval_func的运用
    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5) # verts: spatial coordinates for V unique mesh vertices. faces: triangular faces (each face has exactly three indices). normals: normal direction at each vertec. values: maximum value of the data in the local region near each vertex.
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w') # 打开mesh路径

    for v in verts: # 记录verts
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces: # 记录faces
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close() #关闭


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w') #打开mesh路径

    for idx, v in enumerate(verts):
        c = colors[idx] # 第idx个点的color值
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces: # f_plus
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w') # 打开mesh路径

    for idx, v in enumerate(verts):
        vt = uvs[idx] # 第idx个点的uv值
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1 # 0,0, 2,2, 1,1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
