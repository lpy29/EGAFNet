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

from .builder import DATASETS
from .transform import Compose
from .defaults import DefaultDataset

# from pointcept.models.controlnet.cn_feat import control_extractor
from sklearn.decomposition import PCA
import torch.nn.functional as F


@DATASETS.register_module()
class WHUALSDataset(DefaultDataset):
    # def __init__(self, num_points=4096, split="train", data_root="data/whu_als",transform=None):
    def __init__(
        self,
        num_points=40960,
        split="train",
        data_root="data/whu_als",
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
        # self.get_class_weight()

        # self.label_to_names = {0: 'Others', 1: 'Ground', 2: 'Vegetation', 3: 'Lowveg', 4: 'Wire',
        #                         5: 'Building', 6: 'Tree', 7: 'Light'}
        self.label_to_names = {0: 'Others', 1: 'Ground', 2: 'Vegetation', 3: 'Lowveg', 4: 'Wire',
                                5: 'Building',  6: 'Light'}
        self.num_classes = len(self.label_to_names)
        self.num_points=num_points
        
        self.file_list = self.get_data_list()
        # self.control = control_extractor()
        # self.get_controlnet_feat()
        
        super().__init__(split=split, data_root=data_root, transform=transform, test_mode=test_mode, test_cfg=test_cfg, **kwargs)

    def get_data_list(self):
        # data_list = []
        # data_list.append('data/WHU/test_split/3722/3722_44_xyz.npy')
        # print(self.data_root)
        if self.split == "train":
           data_path = os.path.join(
                self.data_root, "train_norm"
            )
        elif self.split == "val":
           data_path = os.path.join(
                self.data_root, "test_norm"
            )
        elif self.split == "test":
            data_path = os.path.join(
                self.data_root, "test_norm"
            )
        else:
            raise NotImplementedError
        data_list = glob.glob(os.path.join(data_path,"*/*_xyz.npy"))#glob.glob(os.path.join(data_path, "*"))
        # for file in data_list:
        #     prefix = file[:file.find('_xyz.npy')] 
        #     relative_z_path = prefix + "_relative_z.npy" 
        #     if not os.path.exists(relative_z_path):
        #         print(relative_z_path)
        return data_list
    
    # def pca_feat(self, X, n_components=128):
    #     """
    #     Perform PCA on the input tensor X (C x W x H).
    #     - X: The input tensor to apply PCA on (C x W x H).
    #     - n_components: Number of components to retain after PCA.
    #     - Returns: The tensor after PCA transformation (n_components x W x H).
    #     """
    #     # First, flatten the input tensor
    #     C, W, H = X.shape
    #     X_flat = X.view(-1, C)  # Flatten to (W*H, C)
        
    #     # Convert tensor to numpy for PCA
    #     X_flat_np = X_flat.detach().cpu().numpy()

    #     # Apply PCA
    #     pca = PCA(n_components=n_components)
    #     X_pca = pca.fit_transform(X_flat_np)  # Shape will be (W*H, n_components)
        
    #     X_restore = X_pca.view(-1, W, H)  # Reshape back to (C, W, H)
        
    #     return torch.tensor(X_restore.cuda())  # Convert back to tensor
    # def pca_feat(self, X, n_components = 3):
    #     # x should be c*{any shape}
    #     # conduct normalization
    #     X = X/torch.norm(X,dim=0,keepdim=True)
    #     # fit
    #     X = X.cuda()
    #     c, *size = X.shape
    #     X = X.reshape(c,-1).T
    #     n, c = X.shape
    #     mean = torch.mean(X, axis=0)
    #     X = X - mean
    #     covariance_matrix = 1 / n * torch.matmul(X.T, X)
    #     eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    #     eigenvalues = eigenvalues.real
    #     eigenvectors = eigenvectors.real
    #     idx = torch.argsort(-eigenvalues)
    #     eigenvectors = eigenvectors[:, idx]
    #     proj_mat = eigenvectors[:, 0:n_components]
    #     # project
    #     X = X.matmul(proj_mat).T
    #     X = X.reshape(tuple([-1] + size))
    #     return X.cpu()
    
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
    
    # def get_controlnet_feat(self):
    #     feat = []
    #     path = self.get_data_list() #glob.glob(os.path.join(data_path, "*"))
        
    #     for xyz_path in path:
    #         prefix = xyz_path[:xyz_path.find('_xyz.npy')]
    #         img_path = prefix + "_img.npy"
            
    #         xyz = np.load(xyz_path)
    #         img = np.load(img_path)
            
    #         points_img = np.floor(xyz[:,:2] / 0.5).astype(np.int32)
    #         control_img = img[points_img[:,1].min():points_img[:,1].max()+1,points_img[:,0].min():points_img[:,0].max()+1,0]# 这里需要确认索引的顺序???:只有点云图像互转时才需要反过来？
    #         depth, dpt_feats = self.control.dpt_feature(control_img)

    #         featlist = [dpt_feats[i] for i in self.layer]
            
    #         pcalist_s = []
    #         for i in range(len(featlist)):
    #             sfeat = featlist[i]
    #             sfeat = F.interpolate(sfeat.unsqueeze(0), size=(control_img.shape[0], control_img.shape[1]), mode='bilinear', align_corners=True)
    #             # conduct pca
    #             sfeat = self.pca_feats([sfeat.squeeze(0)],128)
    #             pcalist_s.append(sfeat[0])
            
    #         # for feat in featlist:
    #         #     self.pca_feat(feat,128)
    #         # pca_list = [self.pca_feat(feat,128) for feat in featlist]
                
    #         feat.append(pcalist_s)
    #     self.feat = feat
    #     del self.control 

    def get_data(self, idx):
        xyz_path = self.file_list[idx]
        
       
        # print(filename)
        # data = read_ply(f)
        # coords = np.vstack((data['x'], data['y'], data['z'])).T.copy().astype(np.float32)
        # semantic_label = data['class'].copy()
        # semantic_label = semantic_label.astype('int32')
        prefix = xyz_path[:xyz_path.find('_xyz.npy')]
        filename = prefix.split('/')[-1]
        img_path = prefix + "_img.npy"
        label_path = prefix + "_label.npy"
        # distribution_path = prefix + "_dis.npy" 
        # control_path = os.path.join(self.data_root, "control_net")+"/" + filename +"_control.npy" 
        relative_z_path = prefix + "_relative_z.npy" 
        # control_img_path = os.path.join(self.data_root, "control_img")+"/" + filename +".png" 
        # control_feat = torch.stack(self.feat[idx],dim=0)
        
        xyz = np.load(xyz_path)
        label = np.load(label_path) - 1
        img = np.load(img_path)
        # control_feat_all = np.load(control_path)
        # control_feat = control_feat_all[:,:,:,[0,1,2,3]]
        # distribution = np.load(distribution_path) # 这里还是9通道，包含了为标记点
        # mask = np.ones((distribution.shape[0],distribution.shape[1],1))

        # distribution[distribution[:,:,0] != 0,:] = 0 # 有未分类点的区域全部置0
        # mask[distribution.sum(2)==0] = 0 # 没有点的区域全部置0
        # distribution = distribution[:,:,1:] # 去掉未分类
        # distribution[distribution != 0] = 1 # 忽略点数
        # target_index = (distribution.sum(2)!=0)
        # distribution = distribution.astype(np.float32)
        #distribution[target_index] /= np.sum(distribution[target_index],axis=1,keepdims=True)

        points_img = np.floor(xyz[:,:2] / 0.5).astype(np.int32)
        # control_point_img = points_img - points_img.min(0)
        # lowest_z = img[points_img[:,1],points_img[:,0],1] # 这里需要确认索引的顺序！！！！
        relative_z = np.load(relative_z_path)#xyz[:,2] - lowest_z
        xyz[:,:2] -= xyz[:,:2].mean(0) # 只需要中心化x和y，z的范围相对较小且对分类比较重要
        # # data augmentation
        if self.split == "train":
            # 2D aug
            flip_2d = np.random.choice(6, 1)
            #flip_2d = 999
            if flip_2d == 1:
                #pdb.set_trace()
                img = np.ascontiguousarray(np.fliplr(img))
                # distribution = np.ascontiguousarray(np.fliplr(distribution))
                # mask = np.ascontiguousarray(np.fliplr(mask))
                points_img[:, 0] = img.shape[1] - 1 - points_img[:, 0]
                
            elif flip_2d == 2:
                #pdb.set_trace()
                img = np.ascontiguousarray(np.flipud(img))
                # distribution = np.ascontiguousarray(np.flipud(distribution))
                # mask = np.ascontiguousarray(np.flipud(mask))
                points_img[:, 1] = img.shape[0] - 1 - points_img[:, 1]

            elif flip_2d == 3:
                #pdb.set_trace()
                img = np.ascontiguousarray(np.flipud(img))
                # distribution = np.ascontiguousarray(np.flipud(distribution))
                # mask = np.ascontiguousarray(np.flipud(mask))
                points_img[:, 1] = img.shape[0] - 1 - points_img[:, 1]
                img = np.ascontiguousarray(np.fliplr(img))
                # distribution = np.ascontiguousarray(np.fliplr(distribution))
                # mask = np.ascontiguousarray(np.fliplr(mask))
                points_img[:, 0] = img.shape[1] - 1 - points_img[:, 0]

            ######
            #Rotation
            rot_2d = np.random.choice(4, 1)
            if rot_2d == 1: 
                img = np.ascontiguousarray(np.rot90(img))  # 逆时针
                # distribution = np.ascontiguousarray(np.rot90(distribution))
                # mask = np.ascontiguousarray(np.rot90(mask))
                points_img = np.fliplr(points_img)
                points_img[:, 1] = img.shape[1] - 1 - points_img[:, 1]
                #print("conter clock wise")

            elif rot_2d == 2:
                img = np.ascontiguousarray(np.rot90(img,3))  # 顺
                # distribution = np.ascontiguousarray(np.rot90(distribution,3))
                # mask = np.ascontiguousarray(np.rot90(mask,3))
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
                # control_point_img = np.ascontiguousarray(control_point_img),
                # control_feat = [control_feat],
                img = img.reshape(1, 256, 256, 4),
                segment = label,
                name=filename,
            )

        return data_dict
    
    
    def get_class_weight(self):
        # num_class = np.zeros(self.num_classes)
        #     for i in range(len(self.get_data_list())):
        #         label_tmp = self.get_data_list()[i]
        #         f = Path(label_tmp)
        #         # print(filename)
        #         data = read_ply(f)
        #         semantic_label = data['class'].copy()
        #         label = semantic_label.astype('int32')
        #         lbl, count = np.unique(label, return_counts=True)

        #         num_class[lbl] += count
        #     weights = num_class / sum(num_class)
            # inv_weights = 1 / (weights + 0.0
        #      weights = num_class / sum(num_class)
        # # inv_weights = 1 / (weights + 0.02)
        # inv_weights = 1 / weights ** 0.5
        seg_label_weights = np.zeros(7)
        for test_file_name in self.get_data_list():
            pc_path = test_file_name 
            prefix = pc_path[:pc_path.find('_xyz')]   
            label_path = prefix + "_label.npy"
            labels = np.load(label_path).astype(np.int8)
            labels = labels #- 1 # ignore class 0
            labels = labels[labels != -1]
            for nc in range(7):
                seg_label_weights[nc] += (labels==nc).sum()
                
        # seg_label_weights = [546,180850,193723,4614,12070,152045,27250,47605,135173]
        # seg_label_weights = np.array(seg_label_weights)
        seg_label_weights = seg_label_weights / np.sum(seg_label_weights)
        inv_weights = 1 / seg_label_weights ** 0.5
        # seg_label_weights = np.power(np.amax(seg_label_weights) / seg_label_weights, 1 / 3.0)
        return inv_weights#torch.from_numpy(seg_label_weights).float()
