```
################################
# Install CUDA
# https://askubuntu.com/questions/1288672/how-do-you-install-cuda-11-on-ubuntu-20-10-and-verify-the-installation
################################
# Optional, but it is always good to first remove potential previously installed NVIDIA drivers:
sudo apt-get purge cuda* --fix-missing
sudo apt autoremove

# install latest driver (at the time)
sudo apt install nvidia-driver-525 --fix-missing

# install nvidia cuda toolkit
sudo apt install nvidia-cuda-toolkit
export CUDA_PATH=/usr
source ~/.bashrc
nvidia-smi

# post installation (replace with XX.X version you use)
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


################################
# Install OpenPCDet
# Limitations: I limit the version to below:
# Ubuntu 18
# Python 3.10.4
# PyTorch 2.0 on CUDA 11.7
# OpenPCDet: the one i forked

preliminaries:
1) make sure your GPU can support the CUDA XX.X version
2) make sure you can install pytorch/spconv for this CUDA XX.X version

* in this case, we use CUDA 11.7 version
################################
conda remove -n openpcdet --all

conda create --name openpcdet python=3.10.4 -y
conda activate openpcdet

# make sure youre using CUDA11.7 (if you have multiple version installed)
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# this may not be the download command for your prefered version
# always get pytorch installation command for CUDA 11.7 version from [2] or [3]
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# get SPConv specifically for CUDA 11.7
pip install spconv-cu117

# setup the OpenPCDet repo
git clone https://github.com/yjwong1999/OpenPCDet.git
cd OpenPCDet
python setup.py develop

# other dependencies
pip install numpy==1.25
pip install pandas==1.5.3
pip install plyfile==1.0.1
pip install opencv-python==4.7.0.72
pip install av2==0.2.1
pip install kornia==0.5.8
pip install mayavi==4.8.1
pip install PyQt5==5.15.9
pip install open3d==0.17.0
pip install chardet==5.2.0


################################
# Reference
################################
[1] https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md
[2] https://pytorch.org/
[3] https://pytorch.org/get-started/previous-versions/
[4] https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local
```
