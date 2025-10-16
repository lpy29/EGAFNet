<h2>
<a href="https://whu-usi3dv.github.io/LaneMapping/" target="_blank">A novel projection map driven multimodal fusion framework for ALS point cloud segmentation</a>
</h2>

This is the official PyTorch implementation for JAG 2025 paper:A novel projection map driven multimodal fusion framework for ALS point cloud segmentation



## 🔭 Introduction

<p align="center" style="font-size:18px">
<strong>A novel projection map driven multimodal fusion framework for ALS point cloud segmentation</strong>
</p>
<img src="media/teaser.jpg" alt="Network" style="zoom:10%;">

<p align="justify">
<strong>Abstract:</strong> Semantic segmentation of urban point clouds captured by Airborne Laser Scanning (ALS) is essential for understanding complex 3D environments, serving as a robust underlying data foundation for digital twin applications.
The fusion of multimodal data has been proven to significantly improve the performance of ALS semantic segmentation by fully mining rich complementary information in each modality. However, existing fusion-based ALS semantic segmentation methods face critical limitations due to the reliance on multiple sensors, which constrains their applicability. To this end, we propose a novel multimodal framework Elevation Guidance Adaptive Fused Network, termed EGAFNet, that integrates naturally formed top-view projection images from ALS to enhance the information perception of the point cloud. Specifically, to generate highly discriminative input representation, we propose a novel projection method that accurately preserves the relative height relationships between objects and develop a Height Adaptive Scaling Module (HASM) to adaptively adjust object heights, enhancing the expressive capability of elevation information in the projection images.As for feature representation, we design a dual-branch network that effectively captures local and global context from the projection images within a large receptive field. Meanwhile, we propose an Elevation Guidance Adaptive Fusion Module (EGAFM) that adaptively fuses 2D and 3D features based on occlusion relationships to reduce feature confusion caused by occlusion in elevation projection, ensuring meaningful fusion between multimodal features. Extensive experiments on three public datasets demonstrate that our EGAFNet outperforms current state-of-the-art methods.
</p>

## 🆕 News

- 2025-10-11: our paper is accepted for publication in the International Journal of Applied Earth Observation and Geoinformation(JAG)! 🎉

## 💻 Requirements

The code has been trained on:

- Ubuntu 20.04 and above.
- CUDA 11.3 and above.
- Python 3.8 and above.


## 🔧 Installation

### Create a conda virtual environment and activate it.

```
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# Open3D (visualization, optional)
pip install open3d
```


### Install torchsparse:

```
conda install google-sparsehash -c bioconda
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:CPLUS_INCLUDE_PATH
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```



## 💾 Datasets

We used [WHU-Urban](https://whu3d.com/download.html), [DALES](go.udayton.edu/dales3d.) and [STPLS3D](https://github.com/meidachen/STPLS3D) for training and three datasets for evaluation.



## ⏳ Train

To train the network, prepare the dataset and put it in  './data/.'. Then, you use the following command:

```bash
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```
python tools/train.py --config-file configs/whu_als/semseg_spvcnn_fusion.py --num-gpus 2 --options save_path=log/spvcnn_fusion
```

## ✏️ Test

To evaluate the network, you can use the following commands, and do not forget to modify the corresponding datapath in the config file:

```bash
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}

```


<!-- ## 💡 Citation

If you find this repo helpful, please give us a 😍 star 😍.
Please consider citing **LaneMapping** if this program benefits your project

```Tex
@article{MI2024104139,
title = {A benchmark approach and dataset for large-scale lane mapping from MLS point clouds},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {133},
pages = {104139},
year = {2024},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2024.104139},
url = {https://www.sciencedirect.com/science/article/pii/S156984322400493X},
author = {Xiaoxin Mi and Zhen Dong and Zhipeng Cao and Bisheng Yang and Zhen Cao and Chao Zheng and Jantien Stoter and Liangliang Nan}
}
``` -->

## 🔗 Related Projects

We sincerely thank the excellent projects:

- [Pointcept](https://github.com/Pointcept/Pointcept) for base code framework;
- [FreeReg](https://github.com/WHU-USI3DV/FreeReg)  and [SparseDC](https://github.com/WHU-USI3DV/SparseDC) for readme template.
