"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
# import cv2
import numpy as np
from scipy.stats import binned_statistic_2d
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from tools.vis import write_ply

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage
from pointcept.utils.registry import Registry



TRAINERS = Registry("trainers")

def get_nDSM(data_dict):
    
    batch_nDSM = []
    batch_std = []
    batch_label = []
    batch_proj = []
    
    num_batches = len(data_dict["offset"]) 
    batch_offset = data_dict["offset"]
    batch_offset = np.insert(batch_offset , 0,0)  # 添加结束索引
    for i in range(num_batches): 
        start_idx = batch_offset[i] 
        end_idx = batch_offset[i + 1] 
        coords = data_dict["coord"][start_idx:end_idx][:]
        labels = data_dict["segment"][start_idx:end_idx]
        
        # 假设 sampled_data 是经过格网化采样后的数据 
        # coords = data_dict["coord"] # 定义网格分辨率 
        grid_resolution = 0.5
        # 获取点的最小和最大坐标
        # 提取 x, y, z 坐标
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]

        # 确定 DSM 的范围
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max() 
        
        # 投影点云到 nDSM 图像上
        point_indices = np.zeros((coords.shape[0], 2), dtype=int)  # 记录点云投影后的索引
        
        # 定义网格大小 
        x_size = int((max_x - min_x) / grid_resolution) + 1 
        y_size = int((max_y - min_y) / grid_resolution) + 1
        # 初始化 DSM 和 DTM 
        DSM = np.full((x_size, y_size), -np.inf)
        label_grid = np.full((x_size, y_size), -1)  # 用于存储标签
        # 遍历每个点，更新 DSM 和 DTM
        for i in range(len(coords)):
            point = coords[i] 
            x_idx = int((point[0] - min_x) / grid_resolution) 
            y_idx = int((point[1] - min_y) / grid_resolution) 
            point_indices[i] = [x_idx, y_idx]
            DSM[x_idx, y_idx] = max(DSM[x_idx, y_idx], point[2]) 
            # DTM[x_idx, y_idx] = min(DTM[x_idx, y_idx], point[2]) 
            if label_grid[x_idx, y_idx] == -1 or point[2] > DSM[x_idx, y_idx]:
                label_grid[x_idx, y_idx] = labels[i] 

        # 计算每个格网单元的标准差
        from scipy.stats import binned_statistic_2d
        # H_std, x_edges, y_edges, binnumber = binned_statistic_2d(x_coords, y_coords, z_coords, statistic='std', bins=(grid_resolution, grid_resolution))
        # # 替换初始值 
        DSM[DSM == -np.inf] = np.nan
        # DTM[DTM == np.inf] = DSM[DTM == np.inf] 
        # DSM[np.isnan(DSM)] = np.nanmin(DSM)
        # from scipy.ndimage import gaussian_filter

        # # 用局部平均值填充NaN值，以便能够进行高斯平滑
        # nan_mask = np.isnan(DSM)
        # mean_value = np.nanmean(DSM)  # 全局平均值用于填补空洞
        # DSM[nan_mask] = mean_value

        # # 应用高斯滤波器进行全局平滑
        # sigma = 2  # 标准差，控制平滑程度，可以根据需要调整
        # filled_image = gaussian_filter(DSM, sigma=sigma)
        # 计算 nDSM 
        nDSM = DSM - z_coords.min().numpy()
        
        # 计算高程的标准差
        # 计算高程的标准差 H_std
        H_std, x_edges, y_edges, binnumber = binned_statistic_2d(x_coords.cpu().numpy(), y_coords.cpu().numpy(), z_coords.cpu().numpy(), statistic='std', bins=(x_size, y_size))
        H_std[np.isnan(H_std)] = np.nanmin(H_std)

        # # 计算高程的标准差 H_std
        # H_std = np.sqrt(np.sum((nDSM - H_avg) ** 2) / len(nDSM))
        
        # Min-Max归一化
        # ndsm_normalized = nDSM.copy()
        dsm_min = np.nanmin(nDSM)
        dsm_max = np.nanmax(nDSM)
        ndsm_normalized = (nDSM - dsm_min) / (dsm_max - dsm_min)
        
        from scipy.interpolate import griddata
        # # 创建用于插值的网格
        x, y = np.meshgrid(np.arange(ndsm_normalized.shape[1]), np.arange(ndsm_normalized.shape[0]))

        # # 确定有效和无效点
        valid_mask = np.isfinite(ndsm_normalized)
        points = np.array((x[valid_mask], y[valid_mask])).T
        values = ndsm_normalized[valid_mask]
        
        # # # 使用三次插值方法进行插值
        # filled_image = griddata(points, values, (x, y), method='cubic')
        
        # filled_image[np.isnan(filled_image)] = np.nanmin(filled_image)

        # 检查插值结果中的NaN值，并使用最邻近值进行替换
        if np.isnan(ndsm_normalized).any():
            # 对NaN区域进行处理：使用最近的有效值替换NaN
            filled_image = griddata(points, values, (x, y), method='nearest')

        
        # 设置核大小
        kernel = np.ones((3,3),np.uint8)
        
        # 创建结果矩阵，初始化为label_matrix
        smooth_label = label_grid.copy()
        
        # # 创建一个空白矩阵来存储结果
        # smooth_label = -np.ones_like(label_matrix)

        # for label in [1,2,3,4,6,5,7,5,8]:
        #     # 获取当前类别的掩码
        #     mask = (label_grid == label).astype(np.uint8)
        #     # 检查掩码是否包含非零元素
        #     if np.any(mask):
        #         # label_open = cv2.morphologyEx(mask , cv2.MORPH_OPEN, kernel, iterations=1)
        #         label_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel,iterations=1)
        #         open_close =  cv2.morphologyEx(label_close, cv2.MORPH_CLOSE, kernel,iterations=1)

        #         # 将处理后的结果保存到相应位置
        #         smooth_label[open_close == 1] = label

        # smooth_label[smooth_label == -1] = 0
        batch_nDSM.append(filled_image)
        batch_std.append(H_std)
        batch_label.append(smooth_label)
        batch_proj.append(point_indices)

    # batch_nDSM = np.array(batch_nDSM)
    # batch_label = np.array(batch_label)
    batch_proj = np.vstack(batch_proj)
    data_dict["proj"] = batch_proj
    
    return batch_nDSM, batch_std, batch_label,batch_proj

def get_nDSM2(train_loader):

    nDSM = []
    std = []
    label = []
    proj = []
    
    for i, data in enumerate(train_loader):
        batch_nDSM, batch_std,batch_label, batch_proj = get_nDSM(data)
    
        nDSM.append(batch_nDSM)
        std.append(batch_std)
        label.append(batch_label )
        proj.append(batch_proj)

    return nDSM, std, label, proj



class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        
        self.nDSM, self.zstd,self.twod_label, self.proj = None,None,None,None
        self.test_nDSM,self.test_zstd,self.test_twod_label, self.test_proj = None,None,None,None
     
    def compute_inverse_frequency_weighting(self, nDSM, num_classes):
        # 统计每个类别的频率
        num_class = np.zeros(num_classes)
        for i in range(len(nDSM)):
            for j in range(len(nDSM[i])):
                label_tmp = nDSM[i][j]
                # _, _, semantic_label, _, _ = torch.load(label_tmp)
                label = label_tmp.astype('int32')
                lbl, count = np.unique(label, return_counts=True)

                num_class[lbl] += count
        weights = num_class / sum(num_class)
            # inv_weights = 1 / (weights + 0.02)
        inv_weights = 1 / weights ** 0.5
        return inv_weights
    
    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            # self.logger.info(">>>>>>>>>>>>>>>> Generate ControlNet feat >>>>>>>>>>>>>>>>")
            # data_xyz_list = glob.glob(os.path.join(self.cfg.data_root,"*/*_xyz.npy"))
            # data_img_list = glob.glob(os.path.join(self.cfg.data_root,"*/*_img.npy"))
            # self.nDSM, self.zstd, self.twod_label, self.proj = get_nDSM2(self.train_loader)
            # self.test_nDSM,self.test_zstd,self.test_twod_label, self.test_proj = get_nDSM2(self.val_loader)
            
            # self.logger.info(self.compute_inverse_frequency_weighting(self.twod_label, 7))
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                if(self.epoch < 90):
                    if(self.epoch%20 == 0):
                        self.after_epoch()
                else:
                    self.after_epoch()
            # => after train
            self.after_train()
        
    def run_step(self):
        # image = self.nDSM[self.comm_info["iter"]]
        # zstd = self.zstd[self.comm_info["iter"]]
        # proj = self.proj[self.comm_info["iter"]]
        # twod_label =self.twod_label[self.comm_info["iter"]]
        input_dict = self.comm_info["input_dict"]
        # input_dict["proj"] = torch.tensor(proj)
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        
        # image = []
        # zstd = []
        # avg = []
        # zmin = []
        # twod_label = []
        # for file_name in input_dict["name"]:
        #     ndsm_path = os.path.join(self.cfg.data_root, "train_split_img",file_name+"_ndsm.npy")
        #     # 读入单通道的png图像
        #     img = np.load(ndsm_path)
        #     image.append(img)
        #     std_path = os.path.join(self.cfg.data_root, "train_split_img",file_name+"_std.npy")
        #     std = np.load(std_path)
        #     zstd.append(std)
        #     avg_path = os.path.join(self.cfg.data_root, "train_split_img",file_name+"_avg.npy")
        #     zavg = np.load(avg_path)
        #     avg.append(zavg)
        #     min_path = os.path.join(self.cfg.data_root, "train_split_img",file_name+"_min.npy")
        #     min = np.load(min_path)
        #     zmin.append(min)
            
        #     label_path = os.path.join(self.cfg.data_root, "train_split_img",file_name+"_gt.npy")
        #     label = np.load(label_path)
        #     twod_label.append(label)
            
        
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            torch.cuda.empty_cache()
            self.scaler.scale(loss).backward()
            
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=False,#(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            # collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            collate_fn=partial(point_collate_fn),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
