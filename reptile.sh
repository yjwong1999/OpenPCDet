#!/bin/bash

###############################################################################################
# cleansing unused files/directories
###############################################################################################
# remove all pointrcnn files with reptile name
cd  output/custom_models
find . -type d -name "*reptile*" -exec rm -rf {} +

# remove previous round of pointrcnn to prevent clash
rm -r -v "pointrcnn"

# create a copy of pretrained model
rm -r "pointrcnn_pretrained"
cp -r "pointrcnn_pretrained (copy)" "pointrcnn_pretrained"
cd ../../


###############################################################################################
# now lets start
###############################################################################################
# convert raw data to openpcdet format (this will create data/reptile, data/reptile_2, ...)
if [ ! -d "data/reptile" ]; then
  python3 convert_raw_data_for_reptile.py --dir "data_raw/Mimos Dataset 2023/labels"
fi
count=$(ls -dq *data/reptile* | wc -l)
echo -e "There are $count Reptile Training Dataset \n"

# define hyperparameters
epochs=10
outer_loop=100
inner_loop=$count
epsilon=1.0

# outer loop
for i in $(seq 1 $outer_loop)
do
  # initialize a string to hold all task specific models, which will be argument for python3 reptile.py
  # (i.e.: '"/xxx/xxx/model" "/xxx/xxx/model_2" "/xxx/xxx/model_3" ')
  all_model=''
  
  for j in $(seq 1 $inner_loop)
  do
    # all reptile data are renamed as data/custom for generalizability
    if [ $j -eq 1 ]
    then
        filename="data/reptile"
    else
        filename="data/reptile_$j"
    fi
    echo -e "Dealing with $filename"
    mv $filename "data/custom"
  
    # generate the data using openpcdet
    if [ ! -d "data/custom/gt_database" ]; then
      python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
    fi
  
    # train pointrcnn on this reptile data
    cd tools
    python train.py --cfg_file cfgs/custom_models/pointrcnn.yaml --batch_size 2 --workers 1 --epochs $epochs --pretrained_model ../output/custom_models/pointrcnn_pretrained/reptile.pth
    cd ../
  
    # rename back the file
    mv "data/custom" $filename
  
    # rename the pointrcnn models (to prevent clash and overlap)
    if [ $j -eq 1 ]
    then
        modelname="reptile"
    else
        modelname="reptile_$j"
    fi
  
    # rename ../output/custom_models/pointrcnn to ../output/custom_models/pointrcnn (reptile_x), to prevent overwritten by new weights
    mv "/home/tham/Desktop/mimos/OpenPCDet/output/custom_models/pointrcnn" "/home/tham/Desktop/mimos/OpenPCDet/output/custom_models/pointrcnn_$modelname"
  
    # add path of task specific models to all_model
    all_model+=' '
    all_model+="/home/tham/Desktop/mimos/OpenPCDet/output/custom_models/pointrcnn_$modelname/default/ckpt/checkpoint_epoch_$epochs.pth"
  done
  
  # get reptile model (saved in output/custom_models/pointrcnn_pretrained)
  mv "data/reptile" "data/custom"
  cd tools
  python3 reptile.py --cfg_file cfgs/custom_models/pointrcnn.yaml --ckpts $all_model --epoch_id $epochs --pretrained_model ../output/custom_models/pointrcnn_pretrained/reptile.pth --epsilon $epsilon
  cd ../
  mv "data/custom" "data/reptile"

  # remove all task specific models in previous outer loop
  for j in $(seq 1 $inner_loop)
  do
    if [ $j -eq 1 ]
    then
        modelname="reptile"
    else
        modelname="reptile_$j"
    fi
    model_dir="/home/tham/Desktop/mimos/OpenPCDet/output/custom_models/pointrcnn_$modelname"
    rm -r -v $model_dir
  done
  
done
