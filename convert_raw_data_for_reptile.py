import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
from plyfile import PlyData
import argparse

###########################################################
# global variables
###########################################################
point_cloud_ranges = [[],[],[]]
magnify_factor = 100

# 1. convert intensity in pcd to rgb, then convert back to intensity
###########################################################
# some useful functions
###########################################################
# convert intensity to RGB
def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue

# convert RGB to intensity
def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint

# convert ply to bin
def convert_ply(input_path, output_path):
    # read data
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    # convert to DataFrame
    data_pd = pd.DataFrame(data)
    # convert rgb to intensity 
    data_pd['intensity'] = 0 #data_pd['red'].map(lambda x: x<<16) + data_pd['green'].map(lambda x: x<<8) + data_pd['blue']  
    data_pd = data_pd.drop('red', axis=1)
    data_pd = data_pd.drop('green', axis=1)
    data_pd = data_pd.drop('blue', axis=1)
    # x 100
    data_pd['x'] = data_pd['x'].map(lambda x: x * magnify_factor)
    data_pd['y'] = data_pd['y'].map(lambda x: x * magnify_factor)
    data_pd['z'] = data_pd['z'].map(lambda x: x * magnify_factor)
    # record point cloud range
    global point_cloud_ranges
    point_cloud_ranges[0].append(data_pd['x'].max())
    point_cloud_ranges[1].append(data_pd['y'].max())
    point_cloud_ranges[2].append(data_pd['z'].max())
    point_cloud_ranges[0].append(abs(data_pd['x'].min()))
    point_cloud_ranges[1].append(abs(data_pd['y'].min()))
    point_cloud_ranges[2].append(abs(data_pd['z'].min()))
    # initialize array to store data
    data_np = np.zeros(data_pd.shape, dtype=float)
    # read names of properties
    property_names = list(data_pd.columns)
    # read data by property
    for i, name in enumerate(
            property_names): 
        data_np[:, i] = data_pd[name]
    # save
    if output_path.endswith('.bin'):
        data_np.astype(float).tofile(output_path)
    elif output_path.endswith('.npy'):
        np.save(output_path, data_np.astype(float), allow_pickle=True)

# convert pcd to bin
def convert_pcd(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
           print(line)

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, required = False, default = 'reptile', help="where is the directory for the labels")
parser.add_argument('--dir', type = str, required = True, help="where is the directory for the labels")
args = parser.parse_args()


###########################################################
# Reptile data statistics
###########################################################
filename = r'/home/tham/Desktop/mimos/OpenPCDet/data_raw/Mimos Dataset 2023/PCA-Preliminary_Performance_Report.xlsx'
sheetname = 'reptile data'

dfs = pd.read_excel(filename, sheet_name=sheetname)

# condition
fail_tall_no_hang = (dfs['fail'] == 'tall') & (dfs['remarks'] != 'hang')
fail_m = (dfs['fail'] == 'm-shape') & (dfs['remarks'] != 'hang')
both_fail_with_hang = (dfs['fail'] == 'both') & (dfs['remarks'] == 'hang')
both_fail_no_hang = (dfs['fail'] == 'both') & (dfs['remarks'].astype(str) == 'nan')

# index
fail_tall_no_hang = list(dfs[fail_tall_no_hang]['index'])
fail_m = list(dfs[fail_m]['index'])
both_fail_with_hang = list(dfs[both_fail_with_hang]['index'])
both_fail_no_hang = list(dfs[both_fail_no_hang]['index'])

fail_tall_with_hang = list(dfs['index'])
fail_tall_with_hang = sorted(list(set(fail_tall_with_hang) - set(fail_tall_no_hang + fail_m + both_fail_with_hang + both_fail_no_hang)))

# stats
print('###########################')
print('Reptile Data STATS')
print('###########################')
print('fail_tall_no_hang\t:', len(fail_tall_no_hang))
print('fail_m\t\t\t:', len(fail_m))
print('both_fail_with_hang\t:', len(both_fail_with_hang))
print('both_fail_no_hang\t:', len(both_fail_no_hang))
print('fail_tall_with_hang\t:', len(fail_tall_with_hang))
print('total data\t\t:', len(fail_tall_no_hang + fail_m + both_fail_with_hang + both_fail_no_hang + fail_tall_with_hang))
print()

# split to k shot
import copy
k = 10000 # big value means all shot
all_fail_tall_no_hang = [copy.deepcopy(fail_tall_no_hang[i:i+k]) for i in range(0, len(fail_tall_no_hang), k)]
all_fail_tall_with_hang = [copy.deepcopy(fail_tall_with_hang[i:i+k]) for i in range(0, len(fail_tall_with_hang), k)]
all_fail_m = [copy.deepcopy(fail_m[i:i+k]) for i in range(0, len(fail_m), k)]

reptile_data = {
                'train' : all_fail_tall_no_hang + all_fail_tall_with_hang + all_fail_m,
                'val'   : [both_fail_with_hang + both_fail_no_hang] 
                }
    
###########################################################
# loop (loop until all Reptile data is created)
###########################################################

for reptile_idx in range(len(reptile_data['train'])):

    ###########################################################
    # prepare the direcotry for new custom data
    ###########################################################
    # prepare a new directory to store the converted data
    new_directory = args.name
    new_directory = os.path.join('data', new_directory)
    for i in range(1, 100, 1):
        temp = f'{new_directory}_{i}'
        if i == 1:
            temp = new_directory
        if not os.path.isdir(temp):
            break
    new_directory = temp
    os.mkdir(new_directory)
    
    # prepare the sub directory for the data
    os.mkdir(os.path.join(new_directory, 'ImageSets'))
    os.mkdir(os.path.join(new_directory, 'points'))
    os.mkdir(os.path.join(new_directory, 'labels'))
    
    
    ###########################################################
    # loop thru all annotation file from labelCloud
    ###########################################################
    # get all data id name
    ids = []
    # read json
    directory = args.dir
    directory = os.path.join(os.getcwd(), directory)

    # preparing training each training dataset
    # actually in alll case, we should include reptile_data['val'][-1], but we save memory 
    filenames = reptile_data['train'][reptile_idx] + reptile_data['val'][-1]
    filenames = list(map(lambda x: 'fail' + str(x) + '.json', filenames))
    for filename_idx in tqdm(range(len(filenames)), desc =f'Preparing Reptile training dataset {new_directory}...'):
        # get filename
        filename = filenames[filename_idx]
        filename = os.path.join(directory, filename)
        if filename.endswith('_classes.json'):
            continue
    
        # open file
        with open(filename) as f:
            data = json.load(f)
    
        ###########################################################
        # prepare point cloud file
        ###########################################################
        # convert the corresponding .ply file to .npy
        ply_file = data['path']
        npy_file = ply_file[:ply_file.find('.ply')] + '.npy'
        npy_file = os.path.split(npy_file)[1]
        npy_file = os.path.join(new_directory, 'points', npy_file) # store to new directory
        npy_file = npy_file.replace('fail', '')
        convert_ply(ply_file, npy_file)

        ###########################################################
        # prepare label
        ###########################################################
        objects = data['objects']
        annotations = []
        for obj in objects:
            x, y, z = obj['centroid']['x'], obj['centroid']['y'], obj['centroid']['z']
            dx, dy, dz = obj['dimensions']['length'], obj['dimensions']['width'], obj['dimensions']['height']
            yaw = obj['rotations']['z']
            category_name = obj['name']
            temp = [x, y, z, dx, dy, dz, yaw, category_name]
            temp = [x * magnify_factor, y * magnify_factor, z * magnify_factor, \
                    dx * magnify_factor, dy  * magnify_factor, dz * magnify_factor, \
                    yaw, category_name]
            annotation = ''
            for i in temp:
                annotation += str(i)
                annotation += ' '
            annotation = annotation[:-1]
            annotations.append(annotation)
            
    
        # annotation filename for this point cloud
        annot_filename = os.path.join(npy_file[:npy_file.find('points')], 'labels', os.path.split(npy_file)[1][:-4]+'.txt')
        annot_filename = annot_filename.replace('fail', '')
        with open(annot_filename, "w") as f:
            f.writelines("%s\n" % annotation for annotation in annotations) # store to new directory
    
    
        ###########################################################
        # get id
        ###########################################################
        ids.append(os.path.split(npy_file)[1][:-4].replace('fail', ''))
    
    
    ###########################################################
    # prepare imagesets
    ###########################################################    
    train_ids = reptile_data['train'][reptile_idx]
    # actually val_ids should be reptile_data['val'][-1] for all, but we want to save memory
    val_ids   = reptile_data['val'][-1]
    
    training_data_use_percentage = 1.0
    total_training_data = int(training_data_use_percentage * len(train_ids))
    
    train_txt = os.path.join(npy_file[:npy_file.find('points')], 'ImageSets', 'train.txt')
    val_txt   = os.path.join(npy_file[:npy_file.find('points')], 'ImageSets', 'val.txt')
    
    with open(train_txt, "w") as f:
        f.writelines("%s\n" % id_ for id_ in train_ids[:total_training_data])
    
    with open(val_txt, "w") as f:
        f.writelines("%s\n" % id_ for id_ in val_ids)
    
    ###########################################################
    # get point cloud range
    ###########################################################
    # Point cloud range along z-axis / voxel_size is 40
    # Point cloud range along x&y-axis / voxel_size is the multiple of 16.
    voxel_size_x, voxel_size_y, voxel_size_z = 0.1, 0.1, 0.15
    multiplier_x, multiplier_y, multiplier_z = 16, 16, 40
    
    # ori point_cloud_ranges  (before condition)
    point_cloud_ranges = [np.max(item) for item in point_cloud_ranges]
    
    # x axis
    for i in range(100):
        temp = i * multiplier_x * voxel_size_x
        if temp > point_cloud_ranges[0]:
            break
    point_cloud_ranges[0] = temp
    
    # y axis
    for i in range(100):
        temp = i * multiplier_y * voxel_size_y
        if temp > point_cloud_ranges[1]:
            break
    point_cloud_ranges[1] = temp
    
    # z axis
    for i in range(100):
        temp = i * multiplier_z * voxel_size_z
        if temp > point_cloud_ranges[0]:
            break
    point_cloud_ranges[2] = temp
    
    # [xmin, ymin, zmin, xmax, ymax, zmax]
    temp = [x * -1 for x  in point_cloud_ranges]
    point_cloud_ranges = temp + point_cloud_ranges
    point_cloud_ranges = [float('{0:.2f}'.format(x)) for x in point_cloud_ranges]
    print(point_cloud_ranges)
    
    # restart point_cloud_ranges
    point_cloud_ranges = [[],[],[]]
