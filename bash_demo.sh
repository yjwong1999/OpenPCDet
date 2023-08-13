#!/bin/bash

################################
# README
# This .sh file is to:
# 1. Run demo
################################


################################
# Setup anaconda3
# https://stackoverflow.com/a/70293309
################################
source ~/anaconda3/bin/activate
conda init bash
echo " "
conda activate openpcdet


################################
# Run demo.py
################################
cd tools

python demo.py --cfg_file cfgs/custom_models/pointrcnn.yaml --ckpt ../output/custom_models/pointrcnn/default/ckpt/checkpoint_epoch_300.pth --data_path "../data/custom/points" --ext .npy

#python demo.py --cfg_file cfgs/custom_models/pointpillar.yaml --ckpt ../output/custom_models/pointpillar/default/ckpt/checkpoint_epoch_300.pth --data_path "../data/custom/points" --ext .npy

# "../data/custom/points/RF 010_R0C20_F_Snap3D_part1.npy"

cd ../
