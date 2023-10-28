#!/bin/bash

################################
# README
# This .sh file is to:
# 1. Convert raw data + labelCloud format -> OpenPCDet raw format for custom data
# 2. Convert OpenPCDet raw format to KITTI
# 3. Train the model
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

MODEL='pointpillar'
EPOCH=100
BS=4

if [ $MODEL == "pointpillar" ]; then
    CFG_FILE='tools/cfgs/custom_models/pointpillar.yaml'
elif [ $MODEL == "pointrcnn" ]; then
    CFG_FILE='tools/cfgs/custom_models/pointrcnn.yaml'
elif [ $MODEL == "pv_rcnn" ]; then
    CFG_FILE='tools/cfgs/custom_models/pv_rcnn.yaml'
else
    echo "model type not implemented, please check in hyperparameters"
    exit 1
fi

################################
# Fix the label path name in the json label, in case multiple people did the labelling -> insonsistency in root directory
################################
python3 batch_fix_label.py --ply_dir $PLY_DIR --label_dir $LABEL_DIR


################################
# Convert raw data + labelCloud format -> OpenPCDet raw format for custom data 
################################
if [ -d "data/custom" ]; then
    rm -r "data/custom"
fi
python convert_raw_data.py --name $NAME --dir $LABEL_DIR --cfg_file $CFG_FILE
echo ""

################################
# Convert raw data OpenPCDet raw format for custom data -> some internal format
################################
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml


################################
# Train
################################
# pointrcnn
cd tools
python train.py --cfg_file ${CFG_FILE:6:1000} --epochs $EPOCH --batch_size $BS --workers 1  #--pretrained_model ../output/pretrained_models/pretrained_pointrcnn.pth

################################
# deactivate conda environment
################################
conda deactivate
