import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch 
from PIL import Image
import einops
from copy import deepcopy

# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:

        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):
    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

            # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

        # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def find_min_max(points_xyz):
    x_min = np.min(points_xyz[:, 0])
    x_max = np.max(points_xyz[:, 0])

    y_min = np.min(points_xyz[:, 1])
    y_max = np.max(points_xyz[:, 1])

    z_min = np.min(points_xyz[:, 2])
    z_max = np.max(points_xyz[:, 2])

    return x_min, x_max, y_min, y_max, z_min, z_max


def record_num_points(count_block, num_block_points, record_file, flag_x):
    row = "block_{:3d} points_{:5d};".format(count_block, num_block_points)
    with open(record_file, "a+") as f:
        f.write(row)
        f.write("\t")
        if flag_x:
            f.write("\n")


def record_coordinates(count_block, start_x, start_y, end_x, end_y, record_file, flag_x):
    row = "block_{:3d} coordinate: ({}, {}, {}, {});".format(count_block, start_x, start_y, end_x, end_y)
    with open(record_file, "a+") as f:
        f.write(row)
        f.write("\t")
        if flag_x:
            f.write("\n")


def plot_3d_linear_rect(ax, start_x, start_y, end_x, end_y, points_data, default_z, color='red'):
    """
    画出三维长方体
    :param ax: 
    :param start_x: x方向起始点坐标
    :param start_y: y方向起始点坐标
    :param end_x: 
    :param end_y: 
    :param points_data: 
    :param default_z: 
    :param color: 
    :return: 
    """
    x = start_x
    dx = end_x - start_x
    y = start_y
    dy = end_y - start_y

    num_points = points_data.shape[0]
    if num_points > 0:
        z = np.min(points_data[:, 2])
        dz = np.max(points_data[:, 2]) - np.min(points_data[:, 2])
    else:
        z = default_z[0]
        dz = default_z[1] - default_z[0]

    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]
    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xx, yy, [z], **kwargs)
    ax.plot3D(xx, yy, [z + dz], **kwargs)
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)
    
def vis(feat,fig_name):
    # #vis    
                # # 提取特征图
    feature_map =  feat.squeeze(0).permute(1,2,0)  # 删除batch维度

                # # 将特征图展平为二维数组
    feature_map_flat = feature_map.reshape(-1, feature_map.size(2)).detach().cpu().numpy()

                # # 数据标准化
    scaler = StandardScaler()
    feature_map_scaled = scaler.fit_transform(feature_map_flat)

                # # 进行PCA，保留3个组件
    pca = PCA(n_components=3)
    feature_map_pca = pca.fit_transform(feature_map_scaled)

                # # 将PCA后的结果映射回0-255范围，并转换为整数
    feature_map_pca_normalized = (feature_map_pca - feature_map_pca.min()) / (feature_map_pca.max() - feature_map_pca.min())
    feature_map_pca_normalized = (feature_map_pca_normalized * 255).astype(np.uint8)

                # # 将PCA结果重新塑形为原始特征图的空间维度
    feature_map_pca_reshaped = feature_map_pca_normalized.reshape(feature_map.size(0), feature_map.size(1), 3)
    plt.imsave(fig_name+'.png',feature_map_pca_reshaped)


def pca_feat( X, n_components = 3):
    
    # x should be c*{any shape}
    # conduct normalization
    X = X.float()
    X = X/torch.norm(X,dim=0,keepdim=True)
    # fit
    X = X.cuda()
    c, *size = X.shape
    X = X.reshape(c,-1).T
    n, c = X.shape
    mean = torch.mean(X, axis=0)
    X = X - mean
    covariance_matrix = 1 / n * torch.matmul(X.T, X)
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix.float())
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    idx = torch.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    proj_mat = eigenvectors[:, 0:n_components]
    # project
    X = X.matmul(proj_mat).T
    X = X.reshape(tuple([-1] + size))
    return X.cpu()

def visualize_chw_feat(imgin):
    # input should be c*h*w feature map
    assert len(imgin.shape) == 3        
    img = deepcopy(imgin)
    if img.shape[0] > 3:
        img = pca_feat(img)
    img = einops.rearrange(img, 'c h w -> h w c').clone()

    # normalize to 0-255
    def visual_normalize(img):
        if isinstance(img, torch.Tensor):
            # 如果 img 是 PyTorch 张量，使用 .to() 方法进行转换
            img = img.to(torch.float64)  # 转换为 float64
            # vmin = torch.quantile(img, 0.02)
            # vmax = torch.quantile(img, 0.85)
            vmin = torch.min(img)
            vmax = torch.max(img)
            img = img - vmin
            img = img / (vmax - vmin)
            img = (img * 255.0).clamp(0, 255).to(torch.uint8)  # 转换为 0-255 范围并转为 uint8
            return img.cpu().numpy()  # 转换为 NumPy 数组以便使用 PIL 显示

        elif isinstance(img, np.ndarray):
            # 如果 img 是 NumPy 数组，则继续使用 astype()
            img = img.astype(np.float64)
            # vmin = np.percentile(img, 2)
            # vmax = np.percentile(img, 85)
            vmin = np.min(img)
            vmax = np.max(img)
            img -= vmin
            img /= (vmax - vmin)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            return img
            # img = img.astype(np.float64)
            # vmin = np.percentile(img, 2)
            # vmax = np.percentile(img, 85)
            # img -= vmin
            # img /= vmax - vmin
            # img = (img * 255.0).clip(0, 255).astype(np.uint8)
            # return img

    img = visual_normalize(img)
    img = Image.fromarray(img)
    img.show()
    return img # 将结果存储到字典中