################################
# Get ready the data for annotation
################################
# refer the .sh script  for details
bash bash_data.sh

################################
# Annotate the data
# refer <links to be filled> to use labelCloud efficiently
################################
# activate the conda environment (mimos) designed for annotation
conda activate mimos

# assuming you are in OpenPCDet directory, go to the raw data directory
cd data_raw/techpartnerfile

# run labelCloud software
labelCloud

# after the GUI poped out
# refer <links to be  filled to use labelCloud for annotation

# deactivate the conda environment (mimos)
conda deactivate
cd ../../
