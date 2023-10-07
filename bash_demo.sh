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
# hyperparameters
################################
NAME="custom"
LABEL_DIR="data_raw/techpartnerfile/techpartnerfile_label"
PLY_DIR="data_raw/techpartnerfile/preprocessed_techpartnerfile-ply"

#CFG_FILE='tools/cfgs/custom_models/pointrcnn.yaml'
CFG_FILE='tools/cfgs/custom_models/pointpillar.yaml'
#CFG_FILE='tools/cfgs/custom_models/pv_rcnn.yaml'


################################
# Run demo.py
################################
cd tools

python demo.py --cfg_file ${CFG_FILE:6:1000} --ckpt ../output/custom_models/pointrcnn/default/ckpt/checkpoint_epoch_100.pth --data_path "../data/custom/points" --ext .npy

# "../data/custom/points/RF 010_R0C20_F_Snap3D_part1.npy"

cd ../
