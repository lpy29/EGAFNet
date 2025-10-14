"""
ISPRS3D Part Dataset (Unmaintained)

get processed shapenet part dataset
at "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob
from collections.abc import Sequence
import pickle
import torch
from sklearn.neighbors import KDTree
from pathlib import Path
import einops
from PIL import Image
from copy import deepcopy

from .builder import DATASETS
import einops
from PIL import Image
from copy import deepcopy

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose

# from pointcept.models.controlnet.cn_feat import control_extractor
from sklearn.decomposition import PCA
import torch.nn.functional as F

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    # my adding for kitti process
    if W > 2.5*H: W = int(2.5*H)
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

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


@DATASETS.register_module()
class DALESDataset(DefaultDataset):
    def __init__(
        self,
        num_points=40960,
        split="train",
        data_root="data/dales/train_split",
        transform=None,
        test_mode=False,
        test_cfg=None,
        **kwargs,
    ):
        self.data_root = data_root
        self.split = split
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.transform = Compose(transform)
        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            # self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # self.label_to_names = {0: 'Others', 1: 'Ground', 2: 'Vegetation', 3: 'Lowveg', 4: 'Wire',
        #                         5: 'Building', 6: 'Tree', 7: 'Light'}
        self.label_to_names = {0: 'unknown',
                               1: 'Ground',
                               2: 'Vegetation',
                               3: 'Cars',
                               4: 'Trucks',
                               5: 'Power lines',
                               6: 'Fences',
                               7: 'Poles',
                               8: 'Buildings'}
        self.num_classes = len(self.label_to_names)
        
        self.train_list=[
            '5080_54435',
            '5190_54400',
            '5105_54460',
            '5130_54355',
            '5165_54395',
            '5185_54390',
            '5180_54435',
            '5085_54320',
            '5100_54495',
            '5110_54320',
            '5140_54445',
            '5105_54405',
            '5185_54485',
            '5165_54390',
            '5145_54460',
            '5110_54460',
            '5180_54485',
            '5150_54340',
            '5145_54405',
            '5145_54470',
            '5160_54330',
            '5135_54495',
            '5145_54480',
            '5115_54480',
            '5110_54495',
            '5095_54440'
        ]
        self.val_list = [
            '5145_54340',
            '5095_54455',
            '5110_54475'
        ]
        
        data_list = glob.glob(os.path.join(self.data_root, "train_split","*"))
        self.train_data_list = []
        self.val_data_list = []
        
        for filename in data_list:
            # 检查文件是否以train_list中的任何前缀开头
            if any(filename.split('/')[-1].startswith(prefix) for prefix in self.train_list):
                self.train_data_list.append(filename)
            elif any(filename.split('/')[-1].startswith(prefix) for prefix in self.val_list):
                self.val_data_list.append(filename)
                
        self.num_points=num_points
        
        # self.layer=[0,4,6,8]
        # self.control = control_extractor()
        # self.get_controlnet_feat()
        
        super().__init__(split=split, data_root=data_root, transform=transform, test_mode=test_mode, test_cfg=test_cfg, **kwargs)

    def get_data_list(self):
       
        if self.split == "train":
           data_path = os.path.join(
                self.data_root, "train_6"
            )
        elif self.split == "val":
           data_path = os.path.join(
                self.data_root, "test_6"
            )
        elif self.split == "test":
            data_path = os.path.join(
                self.data_root, "test_6"
            )
        else:
            raise NotImplementedError
        data_list = glob.glob(os.path.join(data_path,"*/*_xyz.npy"))#glob.glob(os.path.join(data_path, "*"))
        return data_list
        filtered_files = []
        id = 0
        xx = 399
        import re
        # 筛选出符合条件的文件
        for path in data_list:
            file_name = os.path.basename(path)
            # 提取文件名中的 id 值
            match = re.search(r'\d+_\d+_(\d+)_', file_name)
            if match:
                id_value = int(match.group(1))
                # 计算行号和列号
                row, col = self.id_to_row_col(id_value, 20)
                # 检查行列条件
                if row in range(3,17) and col in range(3,17):
                    if id_value > id:
                        id = id_value
                    if id_value < xx:
                        xx = id_value
                    filtered_files.append(path)
        
        return filtered_files
        # return data_list  
             
    def id_to_row_col(self,id_value, num_cols):
        """根据 id 和列数计算行号和列号"""
        row = id_value // num_cols
        col = id_value % num_cols
        return row, col
    
    def pca_feat(self, X, n_components = 3):
        # x should be c*{any shape}
        # conduct normalization
        X = X/torch.norm(X,dim=0,keepdim=True)
        # fit
        X = X.cuda()
        c, *size = X.shape
        X = X.reshape(c,-1).T
        n, c = X.shape
        mean = torch.mean(X, axis=0)
        X = X - mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        # project
        X = X.matmul(proj_mat).T
        X = X.reshape(tuple([-1] + size))
        return X.cpu()
    
    # def pca_feats(self, xlist, n_components = 256):
    #     # [tensor: c*h*w]
    #     hws = []
    #     split = [0]
    #     flatten_xlist = []
    #     outlist = []
    #     for item in xlist:
    #         # the final output feature shape
    #         c,*size = item.shape
    #         hws.append(tuple([n_components] + size))
    #         item = item.reshape(c,-1)
    #         split.append(split[-1]+item.shape[1])
    #         flatten_xlist.append(item)
    #     flatten_x = torch.cat(flatten_xlist,dim=1)
    #     feat = self.pca_feat(flatten_x, n_components)
    #     # reshape back
    #     for i in range(len(xlist)):
    #         feat_i = feat[:,split[i]:split[i+1]]
    #         feat_i = feat_i.reshape(hws[i])
    #         outlist.append(feat_i)
    #     return outlist
    
    # def pca_feats(self, feat, n_components = 128):
    #     C, W, H = feat.shape
    #     # 将特征图展平为 (C, W*H)
    #     flatten_x = feat.reshape(C, -1).cpu().numpy()  # 变为 (C, W*H)
        
    #     # feat = self.pca_feat(flatten_x, n_components)
    #     from sklearn.manifold import TSNE

    #     # 使用 t-SNE 映射到 2 维
    #     tsne = TSNE(n_components=n_components, random_state=42)
    #     feat = tsne.fit_transform(flatten_x.T).T 

    #     # print("原始数据形状:", X.shape)         # 输出: (1000, 50)
    #     # print("降维后数据形状:", X_embedded.shape)  # 输出: (1000, 2)


    #     # # 使用 PCA 对通道进行降维
    #     # pca = PCA(n_components=n_components)
    #     # reduced_feature_map = pca.fit_transform(feature_map_flat.T).T   # 输出形状: (C, n_components)
    #     # 将恢复的特征重新 reshape 为 (C, W, H)
    #     restored_feature_map = feat.reshape(-1, W, H)
    #     return restored_feature_map
    
    # def get_controlnet_feat(self):
    #     feat = []
    #     path = self.get_data_list() #glob.glob(os.path.join(data_path, "*"))
        
    #     for xyz_path in path:
    #         prefix = xyz_path[:xyz_path.find('_xyz.npy')]
    #         filename = prefix.split('/')[-1]
    #         img_path = prefix + "_img.npy"
            
    #         # if os.path.exists('data/DALES/control_net/'+filename+'_control.npy'):
    #         #     continue
            
    #         print(filename)
    #         xyz = np.load(xyz_path)
    #         img = np.load(img_path)
            
    #         points_img = np.floor(xyz[:,:2] / 0.5).astype(np.int32)
    #         control_img = img[:,:,0]# 这里需要确认索引的顺序???:只有点云图像互转时才需要反过来？
    #         depth, dpt_feats = self.control.dpt_feature(control_img)
            
    #         featlist = [dpt_feats[i] for i in self.layer]
            
    #         pcalist_s = []
    #         for i in range(len(featlist)):
    #             sfeat = featlist[i]
    #             sfeat = F.interpolate(sfeat.unsqueeze(0), size=(control_img.shape[0], control_img.shape[1]), mode='bilinear', align_corners=True)
    #             # conduct pca
    #             # sfeat = self.pca_feats([sfeat.squeeze(0)],128)
    #             sfeat = self.pca_feats(sfeat,128)
    #             pcalist_s.append(sfeat)
            
               
    #         # feat_img = np.zeros((128,control_img.shape[0],control_img.shape[1],4)).astype(np.float32)
    #         # feat_img[:,:,:,0] = pcalist_s[0]
    #         # feat_img[:,:,:,1] = pcalist_s[1]
    #         # feat_img[:,:,:,2] = pcalist_s[2]
    #         # feat_img[:,:,:,3] = pcalist_s[3]    
    #         # np.save('data/DALES/control_net/'+filename+'_control.npy',feat_img)
    #             torch.cuda.empty_cache()
    #             feat.append(pcalist_s)
    #     self.feat = feat
        # del self.control ,dpt_feats,feat
        
    def load_rgb(self, rgb_fn):
        rgb = np.array(Image.open(rgb_fn)).astype(np.uint8) #0-255
        img = deepcopy(rgb)
        img = HWC3(img)
        # img = resize_image(img, 256)
        return rgb, img

    def sd_input(self, img):
        # rgb loading
        self.H, self.W, self.C = img.shape
        img = (torch.from_numpy(np.array(img).astype(np.float32))-127.5)/ 127.5  # must be [-1,1]
        # img = einops.rearrange(img[None], 'b h w c -> b c h w').clone()
        # img = img.to(self.capturer.device)
        return img
    
    def get_data(self, idx):
        xyz_path = self.get_data_list()[idx]
        
        # print(filename)
        # data = read_ply(f)
        # coords = np.vstack((data['x'], data['y'], data['z'])).T.copy().astype(np.float32)
        # semantic_label = data['class'].copy()
        # semantic_label = semantic_label.astype('int32')
        prefix = xyz_path[:xyz_path.find('_xyz.npy')]
        filename = prefix.split('/')[-1]
        img_path = prefix + "_img.npy"
        label_path = prefix + "_label.npy"
        distribution_path = prefix + "_dis.npy" 
        relative_z_path = prefix + "_relative_z.npy" 
        control_path = os.path.join(self.data_root, "control_feat")+"/" + filename +"_control.npy" 
        control_img_path = os.path.join(self.data_root, "control_img")+"/" + filename +".png" 

        # 获取不连续的元素
        xyz = np.load(xyz_path)
        label = np.load(label_path) - 1
        img = np.load(img_path)
        W,H,_ = img.shape
        if W!=256 or H!=256:
            print(filename)
        # sd input
        # _, control_img = self.load_rgb(control_img_path)
        # control_img = self.sd_input(control_img)
        # img = np.concatenate((img_depth, control_img), axis=-1)
        # img = img[..., [0,2, 3]]
        # control_feat_all = np.load(control_path)
        # control_feat = control_feat_all#[:,:,:,[2, 3, 4, 5]]
        #control_img = control_feat_all[:,:,:,[2, 3, 4, 5]]
        distribution = np.load(distribution_path) # 这里还是9通道，包含了为标记点
        mask = np.ones((distribution.shape[0],distribution.shape[1],1))

        distribution[distribution[:,:,0] != 0,:] = 0 # 有未分类点的区域全部置0
        mask[distribution.sum(2)==0] = 0 # 没有点的区域全部置0
        distribution = distribution[:,:,1:] # 去掉未分类
        distribution[distribution != 0] = 1 # 忽略点数
        target_index = (distribution.sum(2)!=0)
        distribution = distribution.astype(np.float32)
        #distribution[target_index] /= np.sum(distribution[target_index],axis=1,keepdims=True)

        points_img = np.floor(xyz[:,:2] / 0.5).astype(np.int32)
        control_point_img = points_img - points_img.min(0)
        lowest_z = img[points_img[:,1],points_img[:,0],1] # 这里需要确认索引的顺序！！！！
        relative_z = np.load(relative_z_path)
        xyz[:,:2] -= xyz[:,:2].mean(0) # 只需要中心化x和y，z的范围相对较小且对分类比较重要
        # data augmentation
        if self.split == "train":
            # 2D aug
            flip_2d = np.random.choice(6, 1)
            #flip_2d = 999
            if flip_2d == 1:
                #pdb.set_trace()
                img = np.ascontiguousarray(np.fliplr(img))
                distribution = np.ascontiguousarray(np.fliplr(distribution))
                mask = np.ascontiguousarray(np.fliplr(mask))
                points_img[:, 0] = img.shape[1] - 1 - points_img[:, 0]

            elif flip_2d == 2:
                #pdb.set_trace()
                img = np.ascontiguousarray(np.flipud(img))
                distribution = np.ascontiguousarray(np.flipud(distribution))
                mask = np.ascontiguousarray(np.flipud(mask))
                points_img[:, 1] = img.shape[0] - 1 - points_img[:, 1]

            elif flip_2d == 3:
                #pdb.set_trace()
                img = np.ascontiguousarray(np.flipud(img))
                distribution = np.ascontiguousarray(np.flipud(distribution))
                mask = np.ascontiguousarray(np.flipud(mask))
                points_img[:, 1] = img.shape[0] - 1 - points_img[:, 1]
                img = np.ascontiguousarray(np.fliplr(img))
                distribution = np.ascontiguousarray(np.fliplr(distribution))
                mask = np.ascontiguousarray(np.fliplr(mask))
                points_img[:, 0] = img.shape[1] - 1 - points_img[:, 0]

            ######
            #Rotation
            rot_2d = np.random.choice(4, 1)
            if rot_2d == 1: 
                img = np.ascontiguousarray(np.rot90(img))  # 逆时针
                distribution = np.ascontiguousarray(np.rot90(distribution))
                mask = np.ascontiguousarray(np.rot90(mask))
                points_img = np.fliplr(points_img)
                points_img[:, 1] = img.shape[1] - 1 - points_img[:, 1]
                #print("conter clock wise")

            elif rot_2d == 2:
                img = np.ascontiguousarray(np.rot90(img,3))  # 顺
                distribution = np.ascontiguousarray(np.rot90(distribution,3))
                mask = np.ascontiguousarray(np.rot90(mask,3))
                points_img = np.fliplr(points_img)
                points_img[:, 0] = img.shape[1] - 1 - points_img[:, 0]
                #print("clock wise")

            # 3D aug
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)


            flip_3d = np.random.choice(4, 1)
            if flip_3d == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_3d == 2:
                xyz[:, 1] = -xyz[:, 1]
        
        data_dict = dict(
                coord = xyz,
                relative_z = relative_z,
                points_img = np.ascontiguousarray(points_img),
                img = img.reshape(1, 256, 256, 4),
                # control_img = control_img_path,
                # control_point_img = np.ascontiguousarray(control_point_img),
                # control_feat = [control_feat],#np.expand_dims(control_feat, axis=0),
                segment = label,
                name=self.get_data_name(idx),
            )

        return data_dict
    
    # def get_data(self, idx):
    #     filename = self.get_data_list()[idx]
    #     f = Path(filename)
    #     file_name_, _ = os.path.splitext(filename.split('/')[-1])
    #     # print(filename)
    #     data = read_ply(f)
    #     coords = np.vstack((data['x'], data['y'], data['z'])).T.copy().astype(np.float32)
    #     semantic_label = data['class'].copy()
    #     semantic_label = semantic_label.astype('int32')
        
    #     # self.get_nDSM(coords,semantic_label)
    #     # all_points_idxs = coords.shape[0]
    #     # if all_points_idxs > self.num_points:
    #     #     # 随机选择 num_samples 个索引
    #     #     selected_points_idxs = np.random.choice(all_points_idxs, self.num_points, replace=False)
    #     # # if all_points_idxs < self.num_points:
    #     # #     selected_points_idxs = np.random.choice(all_points_idxs, self.num_points, replace=True) 
    #     # # else:
    #     # #     seed = np.random.choice(all_points_idxs, 1)
    #     # #     center_point = coords[seed][:, 0:3]
    #     # #     search_tree = KDTree(xyz[:, 0:3], leaf_size=10)
    #     # #     _, ind = search_tree.query(center_point, k=self.num_points)
    #     # #     selected_points_idxs = ind[0]
            
        
    #     #     points = coords[selected_points_idxs, 0:3].astype(np.float32)
    #     #     # rgb = rgb[selected_points_idxs, 0:3].astype(np.float32)
    #     #     labels = semantic_label[selected_points_idxs].astype(np.int64)

    #     #     # current_points = np.ones((self.num_points, 6), dtype=np.float32)        # XYZxyzI1
    #     #     # current_points[:, 0:3] = points

    #     #     # center = np.mean(points, axis=0)[:3]
    #     #     # current_points[:, 0:3] -= center


    #     #     data_dict = dict(
    #     #         coord = points,
    #     #         # strength=current_points[:, 6],
    #     #         segment = labels,
    #     #         name=self.get_data_name(idx),
    #     #     )
            
    #     # else:
    #     data_dict = dict(
    #             coord = coords,
    #             # strength=current_points[:, 6],
    #             segment = semantic_label,
    #             name=file_name_,
    #         )

    #     return data_dict
    
    
    def get_dataset_weights(self):
        weights_dir = os.path.join(self.data_root,'_weight')
        os.mkdir(weights_dir) if not os.path.exists(weights_dir) else None
        weights_path = os.path.join(weights_dir, 'DALES' + '.pkl')
        if not os.path.exists(weights_path):
            num_class = np.zeros(self.num_classes)
            for i in range(len(self.get_data_list())):
                label_tmp = self.get_data_list()[i]
                f = Path(label_tmp)
                # print(filename)
                data = read_ply(f)
                semantic_label = data['class'].copy()
                label = semantic_label.astype('int32')
                lbl, count = np.unique(label, return_counts=True)

                num_class[lbl] += count
            weights = num_class / sum(num_class)
            # inv_weights = 1 / (weights + 0.02)
            inv_weights = 1 / weights ** 0.5
            with open(weights_path, 'wb') as f:
                pickle.dump(inv_weights, f)
        else:
            print('>>>>>>>>>>>>>>>> Load dataset weights >>>>>>>>>>>>>>>>')
            with open(weights_path, 'rb') as f:
                inv_weights = pickle.load(f)
        return inv_weights
