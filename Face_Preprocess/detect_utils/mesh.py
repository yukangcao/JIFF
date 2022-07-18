import numpy as np

    
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

def save_obj_mesh_3dmm(mesh_path, verts, verts_color, faces):
    file = open(mesh_path, 'w')
#    file.write('mtllib material0.mtl\nusemtl material0\n')
    for v, vt in zip(verts, verts_color):
        file.write('v %.8f %.8f %.8f %.8f %.8f %.8f\n' % (v[0], v[1], v[2], vt[0], vt[1], vt[2]))
    for f in faces:
        f = f + 1
        file.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    file.close()
    
def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

# https://github.com/ratcave/wavefront_reader
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
                l_face_data.append(f)
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
                    l_face_uv_data.append(f)
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
            
            face_data += l_face_data
            face_uv_data += l_face_uv_data
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

    norms = np.array(norm_data)
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


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False, 
                  with_vertex_color=False):
    vertex_data = []
    norm_data = []
    uv_data = []
    vertex_color_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

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
            # coordinates only: v x y z
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            # coordinates & color: v x y z r g b
            if len(values) == 7 :
                vc = list(map(float, values[4:7]))
                # Append 1 for Alpha in RGBA
                #vc = list(map(float, values[4:7]+[1]))
                vertex_color_data.append(vc)
            
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) not in [0, '']:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1
    
    if with_vertex_color and with_normal :
        vertex_colors = np.array(vertex_color_data)
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, vertex_colors, face_normals


    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

        
    return vertices, faces


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


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

# compute tangent and bitangent
def compute_tangent(vertices, faces, normals, uvs, faceuvs):    
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0,1,0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)

    # NOTE: traditional version is below

    # pts_tris = vertices[faces]
    # uv_tris = uvs[faceuvs]

    # W = np.stack([pts_tris[::, 1] - pts_tris[::, 0], pts_tris[::, 2] - pts_tris[::, 0]],2)
    # UV = np.stack([uv_tris[::, 1] - uv_tris[::, 0], uv_tris[::, 2] - uv_tris[::, 0]], 1)
    
    # for i in range(W.shape[0]):
    #     W[i,::] = W[i,::].dot(np.linalg.inv(UV[i,::]))

    # tan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # tan[faces[:,0]] += W[:,:,0]
    # tan[faces[:,1]] += W[:,:,0]
    # tan[faces[:,2]] += W[:,:,0]

    # btan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # btan[faces[:,0]] += W[:,:,1]
    # btan[faces[:,1]] += W[:,:,1]    
    # btan[faces[:,2]] += W[:,:,1]

    # normalize_v3(tan)
    
    # ndott = np.sum(normals*tan, 1, keepdims=True)
    # tan = tan - ndott * normals

    # normalize_v3(btan)
    # normalize_v3(tan)

    # tan[np.sum(np.cross(normals, tan) * btan, 1) < 0,:] *= -1.0

    return tan, btan

if __name__ == '__main__':
    pts, tri, nml, trin, uvs, triuv = load_obj_mesh('/home/ICT2000/ssaito/Documents/Body/tmp/Baseball_Pitching/0012.obj', True, True)
    compute_tangent(pts, tri, uvs, triuv)
