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
from .defaults import DefaultDataset
from sklearn.neighbors import KDTree
import tqdm
from tools.vis import read_ply
import pickle
import time

@DATASETS.register_module()
class SensatUrbanDataset(DefaultDataset):
    def __init__(self, cfg, num_points=32768,use_potentials=True, split="train", data_root="data/SensatUrban",transform=None):
        
        self.config = cfg
        self.data_root = data_root
        self.split = split
        self.num_points=num_points
        self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
                               4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
                               9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}
        self.num_classes = len(self.label_to_names)

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Path of the training files
        self.train_path = "train"

        # List of files to process
        ply_path = os.path.join(self.data_root, self.train_path)

        # Proportion of validation scenes
        self.cloud_names = [
           "cambridge_block_0",
            "cambridge_block_1",
            "cambridge_block_2",
            "cambridge_block_3",
            "cambridge_block_4",
            "cambridge_block_6",
            "cambridge_block_7",
            "cambridge_block_8",
            "cambridge_block_9",
            "cambridge_block_10",
            "cambridge_block_12",
            "cambridge_block_13",
            "cambridge_block_14",
            "cambridge_block_17",
            "cambridge_block_18",
            "cambridge_block_19",
            "cambridge_block_20",
            "cambridge_block_21",
            "cambridge_block_23",
            "cambridge_block_25",
            "cambridge_block_26",
            "cambridge_block_28",
            "cambridge_block_32",
            "cambridge_block_33",
            "cambridge_block_34",
            "birmingham_block_1",
            "birmingham_block_3",
            "birmingham_block_4",
            "birmingham_block_5",
            "birmingham_block_6",
            "birmingham_block_7",
            "birmingham_block_9",
            "birmingham_block_10",
            "birmingham_block_11",
            "birmingham_block_12",
            "birmingham_block_13",
        ]   
        # 37 cloud files
        self.all_splits = [
            # Cambridge
            0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            # Birmingham
            0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        ]
        self.validation_split = 1

        # Number of models used per epoch
        if self.split == "train":
            self.epoch_n = self.config['dataset_params']['train_steps'] * self.config['dataset_params']['train_data_loader']['batch_size'] 
        elif self.split in ["val", "test", "ERF"]:
            self.epoch_n = self.config['dataset_params']['val_steps'] * self.config['dataset_params']['val_data_loader']['batch_size']
        else:
            raise ValueError("Unknown split for Sensat Urban data: ", self.split)


        ################
        # Load ply files
        ################

        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.split == "train":
                if self.all_splits[i] != self.validation_split:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            elif self.split in ["val", "test", "ERF"]:
                if self.all_splits[i] == self.validation_split:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            else:
                raise ValueError("Unknown split for Sensat Urban data: ", self.split)

        if self.split == "train":
            self.cloud_names = [
                f
                for i, f in enumerate(self.cloud_names)
                if self.all_splits[i] != self.validation_split
            ]
        elif self.split in ["val", "test", "ERF"]:
            self.cloud_names = [
                f
                for i, f in enumerate(self.cloud_names)
                if self.all_splits[i] == self.validation_split
            ]

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [
                    torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)
                ]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(
                np.array(self.argmin_potentials, dtype=np.int64)
            )
            self.min_potentials = torch.from_numpy(
                np.array(self.min_potentials, dtype=np.float64)
            )
            # self.argmin_potentials.share_memory_()
            # self.min_potentials.share_memory_()
            # for i, _ in enumerate(self.pot_trees):
            #     self.potentials[i].share_memory_()

            # self.worker_waiting = torch.tensor(
            #     [0 for _ in range(config.input_threads)], dtype=torch.int32
            # )
            # self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(
                np.zeros((2, self.epoch_n), dtype=np.int64)
            )
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()
        
        # self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(split=split, data_root=data_root, transform=transform, test_mode=test_mode, test_cfg=test_cfg, **kwargs)

    def load_subsampled_clouds(self):
        tree_path = os.path.join(self.data_root, 'grid_{:.3f}'.format(self.config['dataset_params']['sub_grid_size']))

        for i, file_path in enumerate(self.cloud_names):
            t0 = time.time()
            cloud_name = file_path
            # cloud_split = 'test' if cloud_name in self.test_file_name else 'validation' if cloud_name in self.val_file_name else 'train'

            kd_tree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # if self.split == 'train':
            #     self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # self.input_trees[self.split].append(search_tree)
            # self.input_colors[self.split].append(sub_colors)
            # self.input_labels[self.split].append(sub_labels)
            # self.input_names[self.split].append(cloud_name)
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(os.path.basename(kd_tree_file), size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print("\nPreparing potentials")

            # Restart timer
            t0 = time.time()

            pot_dl = self.config['dataset_params']['in_radius'] / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = os.path.join(
                    tree_path, "{:s}_KDTree.pkl".format(cloud_name)
                )

                # Check if inputs have already been computed
                if os.path.exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, "rb") as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(
                        sub_points.astype(np.float32), sampleDl=pot_dl
                    )

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, "wb") as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print("Done in {:.1f}s".format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.split in ["val", "test"]:

            print("\nPreparing reprojection indices for testing")

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = os.path.join(tree_path, "{:s}_proj.pkl".format(cloud_name))

                # Try to load previous indices
                if os.path.exists(proj_file):
                    with open(proj_file, "rb") as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data["x"], data["y"], data["z"])).T
                    labels = data["class"]

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    # dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, "wb") as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print("{:s} done in {:.1f}s".format(cloud_name, time.time() - t0))
                
        return
    
    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.epoch_n
    # def get_data_list(self):
    #     if self.split == "train":
    #         data_path =  os.path.join(
    #             self.data_root, "train"
    #         )
    #     elif self.split == "val":
    #         data_path =  os.path.join(
    #             self.data_root, "test"
    #         )
    #     elif self.split == "test":
    #         data_path = os.path.join(
    #             self.data_root, "test"
    #         )
    #     else:
    #         raise NotImplementedError
             
    #     data_list = glob.glob(os.path.join(data_path, "*"))
    #     return data_list

    def get_data(self, idx):
        cloud_ind = int(torch.argmin(self.min_potentials))
        point_ind = int(self.argmin_potentials[cloud_ind])

        pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)
        center_point = np.copy(pot_points[point_ind, :].reshape(1, -1))

        if self.split != "ERF":
            center_point += np.clip(
                np.random.normal(
                    scale=self.config['dataset_params']['in_radius'] / 10, size=center_point.shape
                ),
                -self.config['dataset_params']['in_radius'] / 2,
                self.config['dataset_params']['in_radius'] / 2,
            )
            # center_point +=np.random.normal(scale=self.config['dataset_params']['noise_init'] / 10, size=center_point.shape)
            
       
        pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point, r=self.config['dataset_params']['in_radius'], return_distance=True)
        d2s = np.square(dists[0])
        pot_inds = pot_inds[0]

        if self.split != "ERF":
            tukeys = np.square(1 - d2s / np.square(self.config['dataset_params']['in_radius']))
            tukeys[d2s > np.square(self.config['dataset_params']['in_radius'])] = 0
            self.potentials[cloud_ind][pot_inds] += tukeys
            min_ind = torch.argmin(self.potentials[cloud_ind])
            self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
        self.argmin_potentials[[cloud_ind]] = min_ind
        
        points = np.array(self.input_trees[cloud_ind].data, copy=False)
        input_inds = self.input_trees[cloud_ind].query_radius(
            center_point, r=self.config['dataset_params']['in_radius']
        )[0]

        n = input_inds.shape[0]

        # if n < 2:
        #     failed_attempts += 1
        #     if failed_attempts > 100 * self.config.batch_num:
        #         raise ValueError(
        #             "It seems this dataset only containes empty input spheres"
        #         )
        #     continue

        input_points = (points[input_inds] - center_point).astype(np.float32)
        input_colors = self.input_colors[cloud_ind][input_inds]
        input_labels = self.input_labels[cloud_ind][input_inds]
        # if self.split in ["test", "ERF"]:
        #     input_labels = np.zeros(input_points.shape[0])
        # else:
        #     input_labels = self.input_labels[cloud_ind][input_inds]
        #     # input_labels = np.array([self.label_to_idx[l] for l in input_labels])
        
        all_points_idxs = input_points.shape[0]
        if all_points_idxs < self.num_points:
            selected_points_idxs = np.random.choice(all_points_idxs, self.num_points, replace=True) 
        else:
            seed = np.random.choice(all_points_idxs, 1)
            center_point = input_points[seed][:, 0:3]
            search_tree = KDTree(input_points[:, 0:3], leaf_size=10)
            _, ind = search_tree.query(center_point, k=self.num_points)
            selected_points_idxs = ind[0]
            
        
        points = input_points[selected_points_idxs, 0:3].astype(np.float32)
        rgb = input_colors[selected_points_idxs, 0:3].astype(np.float32)
        labels = input_labels[selected_points_idxs].astype(np.int64)

        current_points = np.ones((self.num_points, 3), dtype=np.float32)        # XYZxyzI1
        current_points[:, 0:3] = points

        center = np.mean(points, axis=0)[:3]
        current_points[:, 0:3] -= center
        
        
        data_dict = dict(
            coord=current_points,
            # color=rgb,
            segment=labels,
        )
        return data_dict
    
    
    
    def get_dataset_weights(self):
        weights_dir = os.path.join(self.data_root,'_weight')
        os.mkdir(weights_dir) if not os.path.exists(weights_dir) else None
        weights_path = os.path.join(weights_dir, 'WHUALS' + '.pkl')
        if not os.path.exists(weights_path):
            num_class = np.zeros(self.num_classes)
            for i in range(len(self.get_data_list())):
                label_tmp = self.get_data_list()[i]
                _, _, semantic_label, _, _ = torch.load(label_tmp)
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

# class SensatUrbanDataset(DefaultDataset):
#     def __init__(self, cfg, num_points=32768,use_potentials=True, split="train", data_root="data/SensatUrban",transform=None):
        
#         self.name = 'SensatUrban'
#         self.cfg = cfg
#         # root_path = cfg['dataset_params']['train_data_loader']['data_path'] + split
#         self.data_root = data_root
#         self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
#                                4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
#                                9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}
#         self.num_classes = len(self.label_to_names)
#         self.label_values = np.sort([k for k, v in self.label_to_names.items()])
#         self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
#         self.ignored_labels = np.array([])

#         # self.all_files = glob.glob(os.path.join(self.data_root, 'train','*.ply'))
#         # np.sort(self.all_files)
#         # self.all_files = self.all_files[:1]
#         # self.all_files.extend(glob.glob(os.path.join(self.path, 'test','*.ply')))
        
#         self.all_files = [
#                           "data/SensatUrban/train/birmingham_block_3.ply",
#                           "data/SensatUrban/train/birmingham_block_6.ply",
#                           "data/SensatUrban/train/birmingham_block_7.ply",
#                           "data/SensatUrban/train/birmingham_block_1.ply",
#                           "data/SensatUrban/train/birmingham_block_5.ply",
#                           "data/SensatUrban/train/birmingham_block_2.ply",
#                           ]

#         self.val_file_name = ['birmingham_block_1',
#                               'birmingham_block_5',
#                               'cambridge_block_10',
#                               'cambridge_block_7']
#         # self.test_file_name = []
#         self.test_file_name = ['birmingham_block_2', 'birmingham_block_8',
#                                'cambridge_block_15', 'cambridge_block_22',
#                                'cambridge_block_16', 'cambridge_block_27']
        
#         self.use_val = True

#         # initialize
#         self.num_per_class = np.zeros(self.num_classes)
#         self.val_proj = []
#         self.val_labels = []
#         self.test_proj = []
#         self.test_labels = []
#         self.possibility = {}
#         self.min_possibility = {}
#         self.input_trees = {'train': [], 'validation': [], 'test': []}
#         self.input_colors = {'train': [], 'validation': [], 'test': []}
#         self.input_labels = {'train': [], 'validation': [], 'test': []}
#         self.input_names = {'train': [], 'validation': [], 'test': []}
#         self.load_subsampled_clouds()
        
#         for ignore_label in self.ignored_labels:
#             self.num_per_class = np.delete(self.num_per_class, ignore_label)
            
#         self.split = split
#         self.num_per_epoch = cfg['dataset_params']['train_steps'] * cfg['dataset_params']['train_data_loader']['batch_size'] if split == 'training' else cfg['dataset_params']['val_steps'] * cfg['dataset_params']['val_data_loader']['batch_size']
#         self.possibility = []
#         self.min_possibility = []
#         for i, tree in enumerate(self.input_colors[split]):
#             self.possibility.append(np.random.rand(tree.shape[0]) * 1e-3)
#             self.min_possibility.append(float(np.min(self.possibility[-1])))
        
#         # self.learning_map = self.get_learning_map(ignore_index)
#         super().__init__(data_root=data_root,split = split,transform=transform)

#     def load_subsampled_clouds(self):
#         tree_path = os.path.join(self.data_root, 'grid_{:.3f}'.format(self.cfg['dataset_params']['sub_grid_size']))

#         for i, file_path in enumerate(self.all_files):
#             t0 = time.time()
#             cloud_name = os.path.basename(file_path)[:-4]
#             cloud_split = 'test' if cloud_name in self.test_file_name else 'validation' if cloud_name in self.val_file_name else 'train'

#             kd_tree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
#             sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

#             data = read_ply(sub_ply_file)
#             sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
#             sub_labels = data['class']

#             if cloud_split == 'train':
#                 self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

#             with open(kd_tree_file, 'rb') as f:
#                 search_tree = pickle.load(f)

#             self.input_trees[cloud_split].append(search_tree)
#             self.input_colors[cloud_split].append(sub_colors)
#             self.input_labels[cloud_split].append(sub_labels)
#             self.input_names[cloud_split].append(cloud_name)

#             size = sub_colors.shape[0] * 4 * 7
#             print('{:s} {:.1f} MB loaded in {:.1f}s'.format(os.path.basename(kd_tree_file), size * 1e-6, time.time() - t0))
    
#     def __len__(self):
#         """
#         The number of yielded samples is variable
#         """
#         return self.num_per_epoch
#     # def get_data_list(self):
#     #     if self.split == "train":
#     #         data_path =  os.path.join(
#     #             self.data_root, "train"
#     #         )
#     #     elif self.split == "val":
#     #         data_path =  os.path.join(
#     #             self.data_root, "test"
#     #         )
#     #     elif self.split == "test":
#     #         data_path = os.path.join(
#     #             self.data_root, "test"
#     #         )
#     #     else:
#     #         raise NotImplementedError
             
#     #     data_list = glob.glob(os.path.join(data_path, "*"))
#     #     return data_list

#     def get_data(self, idx):
#         cloud_idx = int(np.argmin(self.min_possibility))
#         point_ind = np.argmin(self.possibility[cloud_idx])

#         points = np.array(self.input_trees[self.split][cloud_idx].data, copy=False)
#         center_point = points[point_ind, :].reshape(1, -1)

#         # noise = np.random.normal(scale=self.cfg['dataset_params']['noise_init'] / 10, size=center_point.shape)
#         pick_point = center_point # + noise.astype(center_point.dtype)

#         if len(points) < self.cfg['dataset_params']['num_points']:
#             queried_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
#         else:
#             queried_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=self.cfg['dataset_params']['num_points'])[1][0]

#         # queried_idx = DP.shuffle_idx(queried_idx)
#         queried_pc_xyz = points[queried_idx]
#         queried_pc_xyz = queried_pc_xyz - pick_point
#         queried_pc_colors = self.input_colors[self.split][cloud_idx][queried_idx]
#         queried_pc_labels = self.input_labels[self.split][cloud_idx][queried_idx]

#         dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
#         delta = np.square(1 - dists / np.max(dists))
#         self.possibility[cloud_idx][queried_idx] += delta
#         self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

#         if len(points) < self.cfg['dataset_params']['num_points']:
#             queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
#                 DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, self.cfg['dataset_params']['num_points'])

#         # return torch.tensor(queried_pc_xyz, dtype=torch.float32), \
#         #        torch.tensor(queried_pc_colors, dtype=torch.float32), \
#         #        torch.tensor(queried_pc_labels, dtype=torch.int32), \
#         #        torch.tensor(queried_idx, dtype=torch.int32), \
#         #        torch.tensor([cloud_idx], dtype=torch.int32)
        
        
#         data_dict = dict(
#             coord=queried_pc_xyz,
#             # color=rgb,
#             segment=queried_pc_labels,
#         )
#         return data_dict
    
    
    
#     def get_dataset_weights(self):
#         weights_dir = os.path.join(self.data_root,'_weight')
#         os.mkdir(weights_dir) if not os.path.exists(weights_dir) else None
#         weights_path = os.path.join(weights_dir, 'WHUALS' + '.pkl')
#         if not os.path.exists(weights_path):
#             num_class = np.zeros(self.num_classes)
#             for i in range(len(self.get_data_list())):
#                 label_tmp = self.get_data_list()[i]
#                 _, _, semantic_label, _, _ = torch.load(label_tmp)
#                 label = semantic_label.astype('int32')
#                 lbl, count = np.unique(label, return_counts=True)

#                 num_class[lbl] += count
#             weights = num_class / sum(num_class)
#             # inv_weights = 1 / (weights + 0.02)
#             inv_weights = 1 / weights ** 0.5
#             with open(weights_path, 'wb') as f:
#                 pickle.dump(inv_weights, f)
#         else:
#             print('>>>>>>>>>>>>>>>> Load dataset weights >>>>>>>>>>>>>>>>')
#             with open(weights_path, 'rb') as f:
#                 inv_weights = pickle.load(f)
#         return inv_weights