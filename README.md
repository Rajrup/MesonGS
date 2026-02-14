## Setup

- Ubuntu 24.04
- GCC 12.4.0
- CUDA 12.1
- CuDNN 9.17.1

## Install Dependencies

```bash
git clone --recursive https://github.com/ShuzhaoXie/MesonGS.git
cd MesonGS

sudo apt install zip unzip
conda env create --file environment.yml
conda activate mesongs

sudo apt-get install libgl1 -y
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install mkl==2023.2.0
pip install open3d==0.18.0
pip install plyfile tqdm einops scipy trimesh Ninja seaborn loguru pandas mediapy
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install submodules/weighted_distance --no-build-isolation
```

### Environment Variables

* Replace the `MAIN_DIR` in [utils/system_utils.py](utils/system_utils.py) with your dir path.

```bash
cd MesonGS
mkdir -p output
mkdir -p data
mkdir -p exp_data
mkdir -p exp_data/csv
```

### Pre-trained Model

Downloaded pre-trained `mic` scene from [here [68 MB]](https://drive.google.com/file/d/1VqDNh7lHraWrA7uj8Dhw62pyZgr_kzLy/view?usp=drive_link). Then:
Unzip and put the checkpoint directory into the `output` directory. 

```bash
mv mic.zip output/
unzip output/mic.zip -d output/mic/
cd output/mic/
mkdir -p point_cloud/iteration_30000/
mv point_cloud.ply point_cloud/iteration_30000/

mic
├── cameras.json
├── cfg_args
├── input.ply
└── point_cloud
    └── iteration_30000
        └── point_cloud.ply
```

## Data Preparation

### 360_v2 Dataset

Link: https://jonbarron.info/mipnerf360/

```bash
cd /synology/rajrup/MesonGS
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d 360_v2/

wget https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
unzip 360_extra_scenes.zip -d 360_extra_scenes/
mv 360_extra_scenes/* 360_v2/

rm -rf 360_extra_scenes/
rm -rf 360_v2/flowers.txt 360_v2/treehill.txt

cd /home/rajrup/Project/MesonGS
ln -s /synology/rajrup/MesonGS/360_v2 data/
```

### Deep Blending (db) and Tank&Truck (tandt) Dataset

Link: 

```bash
cd /synology/rajrup/MesonGS
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip

cd /home/rajrup/Project/MesonGS
ln -s /synology/rajrup/MesonGS/db data/
ln -s /synology/rajrup/MesonGS/tandt data/
```

### Nerf Synthetic Dataset

Link: https://www.matthewtancik.com/nerf.
Download from [Google Drive](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4).

```bash
mv ~/Downloads/NeRF_Data-20260211T054039Z-1-00*.zip /synology/rajrup/MesonGS
cd /synology/rajrup/MesonGS
unzip NeRF_Data-20260211T054039Z-1-001.zip
unzip NeRF_Data-20260211T054039Z-1-002.zip
unzip NeRF_Data-20260211T054039Z-1-003.zip
rm -rf NeRF_Data-20260211T054039Z-1-00*.zip

cd NeRF_Data
unzip nerf_synthetic.zip
mv nerf_synthetic ../

cd /home/rajrup/Project/MesonGS
ln -s /synology/rajrup/MesonGS/nerf_synthetic data/
```

## Training

- Follow README of Original Gaussian Splatting repo to train the model.

## Compression

```bash
# Set the paths to trained model and data. 
# Set --iterations to 0 for compression without finetuning. 
# Add --skip_post_eval to skip tedious testing process.
bash scripts/mesongs_block.sh 
```