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

# post installation
export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

################################
# Install OpenPCDet
# Limitations: I limit the version to below:
# Ubuntu 18
# Python 3.10.4
# PyTorch 2.0 on CUDA 11.7
# OpenPCDet: the one i forked
################################
conda remove -n openpcdet --all

conda create --name openpcdet python=3.10.4 -y
conda activate openpcdet

# get install command from [2] or [3]
pip3 install torch torchvision torchaudio

# get SPConv specifically for CUDA 11.7
pip install spconv-cu117

# make sure youre using CUDA11.7 (if you have multiple version installed)
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}

# setup the OpenPCDet repo
git clone https://github.com/yjwong1999/OpenPCDet.git
cd OpenPCDet
python setup.py develop

# other dependencies
pip install pandas
pip install plyfile
pip install opencv-python==4.7.0.72
pip install av2
pip install kornia
pip install mayavi
pip install PyQt5
pip install openpyxl

[1] https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md
[2] https://pytorch.org/
[3] https://pytorch.org/get-started/previous-versions/
[4] https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local


