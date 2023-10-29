#!/bin/bash

################################
# README
# This .sh file is to:
# 1. Convert raw data + labelCloud format -> OpenPCDet raw format for custom data
# 2. Convert OpenPCDet raw format to KITTI
# 3. Train the model via reptile
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

NUM_TASK=4                    # predefined reptile tasks
EPOCHS=100                    # training epochs
BS=4                          # batch size
EPSILON=0.01                  # outer learning rate of reptile
OUTER_LOOP=1                  # outer loop
INNER_LOOP=$NUM_TASK          # inner loop = number of tasks

PC_MF=20                      # magnifying factor (MF) to scale up point clouds
DXDY_MF="0.85"                # magnifying factor (MF) to scale down dx dy dimension of labels

if [ $MODEL == "pointpillar" ]; then
    CFG_FILE='tools/cfgs/custom_models/pointpillar.yaml'
    PRETRAINED_MODEL='../output/pretrained/pretrained_pointpillar.pth'
elif [ $MODEL == "pointrcnn" ]; then
    CFG_FILE='tools/cfgs/custom_models/pointrcnn.yaml'
    PRETRAINED_MODEL='../output/pretrained/pretrained_pointrcnn.pth'
elif [ $MODEL == "pv_rcnn" ]; then
    CFG_FILE='tools/cfgs/custom_models/pv_rcnn.yaml'
    PRETRAINED_MODEL='../output/pretrained/pretrained_pv_rcnn.pth'
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
python convert_raw_data.py --name $NAME --dir $LABEL_DIR --cfg_file $CFG_FILE --pc_mf $PC_MF --dxdy_mf $DXDY_MF
echo ""


################################
# Download pretrained models as starting weight for reptile
################################
if [ -d "output/pretrained" ]; then
    echo -e "Pretrained models are downloaded previously\n"
else
    if [ -d "output" ]; then
        echo -e "Output directory exists\n"
    else
         mkdir 'output'
    fi
    mkdir 'output/pretrained'
    cd output/pretrained
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm" -O "pretrained_pointpillar.pth" && rm -rf /tmp/cookies.txt
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-" -O "pretrained_pv_rcnn.pth" && rm -rf /tmp/cookies.txt
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU" -O "pretrained_pointrcnn.pth" && rm -rf /tmp/cookies.txt
    cd ../../
fi


################################
# create reptile data
################################
python3 create_reptile_data.py --val-num 20 --upsample 6


################################
# reptile
################################

# remove previous models, just in case 
if [ -d "output/custom_models" ]; then
    rm -rf 'output/custom_models'
fi

# loop outer loop
for i in $(seq 1 $OUTER_LOOP)
do
    echo -e "###############################"
    echo -e "outer loop $i"
    echo -e "###############################"

    # loop inner loop
    all_model='' # for reptile
    for j in $(seq 1 $NUM_TASK)
    do
        echo -e "inner loop (task) $i.$j"

        # set training data
        cd data/custom/ImageSets
        mv "reptile_train_$j.txt" "train.txt"
        cd ../../../

        # Convert raw data OpenPCDet raw format for custom data -> some internal format
        python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
        
        # train
        cd tools
        python train.py --cfg_file cfgs/custom_models/pointpillar.yaml --batch_size 2 --workers 1 --epochs $EPOCHS --pretrained_model $PRETRAINED_MODEL
        cd ../

        # rename output directory
        mv "output/custom_models/pointpillar" "output/custom_models/pointpillar_$j" 

        # model to be reptile
        all_model+=' '
        all_model+="../output/custom_models/pointpillar_$j/default/ckpt/checkpoint_epoch_$EPOCHS.pth"

        # reset training data
        cd data/custom/ImageSets
        mv "train.txt" "reptile_train_$j.txt"
        cd ../../../
    done

    echo -e "###############################"
    echo -e "Reptile now"
    echo -e "###############################"
    cd tools
    python3 reptile.py --cfg_file cfgs/custom_models/pointpillar.yaml --ckpts $all_model --epoch_id $EPOCHS --pretrained_model $PRETRAINED_MODEL --epsilon $EPSILON
    cd ../

    PRETRAINED_PATH='../output/custom_models/reptile.pth' # after reptile, we have the first reptile model

done
