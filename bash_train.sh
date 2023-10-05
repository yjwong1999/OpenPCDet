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
PLY_DIR="data_raw/techpartnerfile/techpartnerfile-ply"

#CFG_FILE='tools/cfgs/custom_models/pointrcnn.yaml'
CFG_FILE='tools/cfgs/custom_models/pointpillar.yaml'
#CFG_FILE='tools/cfgs/custom_models/pv_rcnn.yaml'


################################
# Fix the label path name in the json label, in case multiple did the labelling -> insonsistency
################################
python3 batch_fix_label.py --label_dir data_raw/techpartnerfile/techpartnerfile_label --ply_dir data_raw/techpartnerfile/preprocessed_techpartnerfile-ply


################################
# Convert raw data + labelCloud format -> OpenPCDet raw format for custom data 
################################
'''
if [ -d "data/custom" ]; then
    echo -e "custom data has been created previously."
else
    python convert_raw_data.py --name $NAME --dir $LABEL_DIR --cfg_file $CFG_FILE
fi
echo ""
'''
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
#python train.py --cfg_file ${CFG_FILE:6:1000}  --batch_size 2 --workers 1 --epochs 300 #--pretrained_model ../output/pretrained_models/pretrained_pointrcnn.pth
python train.py --cfg_file ${CFG_FILE:6:1000}  --batch_size 3 --workers 1 --epochs 100
#python train.py --cfg_file ${CFG_FILE:6:1000}  --batch_size 3 --workers 1 --epochs 100

################################
# deactivate conda environment
################################
conda deactivate
