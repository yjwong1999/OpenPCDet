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
MODEL='pointpillar'
NAME="custom"
LABEL_DIR="data_raw/techpartnerfile/techpartnerfile_label"
PLY_DIR="data_raw/techpartnerfile/preprocessed_techpartnerfile-ply"

EPOCH=100

if [ $MODEL == "pointpillar" ]; then
    CFG_FILE='../output/custom_models/pointpillar/default/pointpillar.yaml'
elif [ $MODEL == "pointrcnn" ]; then
    CFG_FILE='../output/custom_models/pointpillar/default/pointrcnn.yaml'
elif [ $MODEL == "pv_rcnn" ]; then
    CFG_FILE='../output/custom_models/pointpillar/default/pv_rcnn.yaml'
else
    echo "model type not implemented, please check in hyperparameters"
    exit 1
fi

################################
# Run demo.py
################################
cd tools

# loop through all point clouds
python demo.py --cfg_file $CFG_FILE --ckpt ../output/custom_models/$MODEL/default/ckpt/checkpoint_epoch_$EPOCH.pth --data_path "../data/custom/points" --ext .npy

# loop through specific point clouds
# python demo.py --cfg_file $CFG_FILE --ckpt ../output/custom_models/$MODEL/default/ckpt/checkpoint_epoch_$EPOCH.pth --data_path "../data/custom/points/RF 010_R0C20_F_Snap3D_part1.npy" --ext .npy

cd ../
