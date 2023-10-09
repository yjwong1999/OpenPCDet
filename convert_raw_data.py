import os, shutil, json, yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from plyfile import PlyData
import argparse

# 0. convert intensity in pcd to rgb, then convert back to intensity
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
    data_pd['intensity'] = 0
    try:
        data_pd = data_pd.drop('red', axis=1)
        data_pd = data_pd.drop('green', axis=1)
        data_pd = data_pd.drop('blue', axis=1)
    except:
        pass
    # x 100
    data_pd['x'] = data_pd['x'].map(lambda x: x * magnify_factor)
    data_pd['y'] = data_pd['y'].map(lambda x: x * magnify_factor)
    data_pd['z'] = data_pd['z'].map(lambda x: x * magnify_factor)
    # record point cloud range
    global point_cloud_ranges
    point_cloud_ranges[0].append(data_pd['x'].min())
    point_cloud_ranges[1].append(data_pd['y'].min())
    point_cloud_ranges[2].append(data_pd['z'].min())
    point_cloud_ranges[3].append(data_pd['x'].max())
    point_cloud_ranges[4].append(data_pd['y'].max())
    point_cloud_ranges[5].append(data_pd['z'].max())
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

# read cfg file (yaml)
def read_cfg_file(cfg_file):
    with open(cfg_file, "r") as stream:
        try:
            cfg_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg_file
 

# 1. everything start here
###########################################################
# some useful functions
###########################################################
# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, required = False, default = 'custom', help="where is the directory for the labels")
parser.add_argument('--dir', type = str, required = True, help="where is the directory for the labels")
parser.add_argument('--cfg_file', type=str, default=None, required = True, help='specify the config for training')
args = parser.parse_args()

# global variable
point_cloud_ranges = [[],[],[],[],[],[]]
magnify_factor = 20
voxel_size_x, voxel_size_y, voxel_size_z = 0.2, 0.2, 0.15 # default
multiplier_x, multiplier_y, multiplier_z = 16, 16, 16
z_range = 40

# read cfg file
cfg_file = args.cfg_file
cfg = read_cfg_file(cfg_file) # read cfg file

model_name = cfg['MODEL']['NAME']

data_cfg_file = cfg['DATA_CONFIG']['_BASE_CONFIG_']
data_cfg_file = os.path.join('tools', data_cfg_file)
data_cfg = read_cfg_file(data_cfg_file) # read data cfg file
'''
# voxel size
voxel_sizes = data_cfg['DATA_PROCESSOR'][2]['VOXEL_SIZE']
print(voxel_sizes)
voxel_size_x, voxel_size_y, voxel_size_z = voxel_sizes
'''
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

# loop until all conditions are met (refer below: get point cloud range)
while True:
    ###########################################################
    # loop thru all annotation file from labelCloud
    ###########################################################
    # get all data id name
    ids = []
    # read json
    directory = args.dir
    directory = os.path.join(os.getcwd(), directory)

    filenames = os.listdir(directory)
    for filename in tqdm(range(len(filenames)), desc =f'Raw data + label -> OpenPCDet standardized format (in {new_directory})...'):
        # get filename
        filename = filenames[filename]
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
        '''
        # replace failxxx.json to xxx.json
        filename = filename.replace('fail', '')
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        '''
        ###########################################################
        # prepare label
        ###########################################################
        objects = data['objects']
        annotations = []
        for obj in objects:
            x, y, z = obj['centroid']['x'], obj['centroid']['y'], obj['centroid']['z']
            dx, dy, dz = obj['dimensions']['length'], obj['dimensions']['width'], obj['dimensions']['height']
            yaw = obj['rotations']['z']
            category_name = obj['name'].replace('good', 'pass').replace('defect', 'fail')
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
    np.random.seed(123)
    np.random.shuffle(ids)

    """
    Method A: separate by percentage
    """
    train_ids = ids[:int(len(ids)*0.8)]
    val_ids   = ids[int(len(ids)*0.8):]

    """
    Method B: separate by augmentation
    """
    train_ids, val_ids = [], []
    for id_ in ids:
        print(id_)
        if 'augmented' in id_:
            train_ids.append(id_)
        else:
            val_ids.append(id_)

    #train_ids = train_ids + val_ids[:int(len(val_ids) / 2)]
    #val_ids = val_ids[int(len(val_ids) / 2) :]
    
    # txt filename (train-val split)
    train_txt = os.path.join(npy_file[:npy_file.find('points')], 'ImageSets', 'train.txt')
    val_txt   = os.path.join(npy_file[:npy_file.find('points')], 'ImageSets', 'val.txt')

    with open(train_txt, "w") as f:
        f.writelines("%s\n" % id_ for id_ in train_ids)

    with open(val_txt, "w") as f:
        f.writelines("%s\n" % id_ for id_ in val_ids)

    ###########################################################
    # get point cloud range
    ###########################################################
    # refer https://github.com/yjwong1999/OpenPCDet/blob/master/docs/CUSTOM_DATASET_TUTORIAL.md
    # # [xmin, ymin, zmin, xmax, ymax, zmax]
    # CONDITION 1: Point cloud range along z-axis / voxel_size is 40 (only for vocel-based detectors)
    # CONDITION 2: Point cloud range along x&y-axis / voxel_size is the multiple of 16.

    # ori point_cloud_ranges  (before condition)
    point_cloud_ranges = [np.around(np.min(item), decimals = 2) for item in point_cloud_ranges[:3]] + [np.around(np.max(item), decimals = 2) for item in point_cloud_ranges[3:]]
    print(point_cloud_ranges)

    ##############
    # condition 2
    ##############
    for i in range(100):
        temp_x = i * multiplier_x * voxel_size_x
        temp_y = i * multiplier_y * voxel_size_y
        temp_z = i * multiplier_z * voxel_size_z

        sub_condition_x = temp_x > point_cloud_ranges[3] - point_cloud_ranges[0]
        sub_condition_y = temp_y > point_cloud_ranges[4] - point_cloud_ranges[1]
        #sub_condition_z = temp_z > point_cloud_ranges[5] - point_cloud_ranges[2]

        if sub_condition_x and sub_condition_y: #and sub_condition_z:
            break
    # x axis
    point_cloud_ranges[0] = point_cloud_ranges[0]
    point_cloud_ranges[3] = point_cloud_ranges[0] + temp_x
    # y axis
    point_cloud_ranges[1] = point_cloud_ranges[1]
    point_cloud_ranges[4] = point_cloud_ranges[1] + temp_y
    # z axis
    point_cloud_ranges[2] = point_cloud_ranges[2]
    point_cloud_ranges[5] = point_cloud_ranges[2] + temp_z

    # round to 2 decimal places
    point_cloud_ranges = [np.around(item, decimals=2) for item in point_cloud_ranges]
    
    ##############
    # condition 1
    ##############
    if model_name in ['PVRCNN', 'SECONDNet']:
        z_min = point_cloud_ranges[2]
        z_max = point_cloud_ranges[5]
        if (z_max - z_min) / voxel_size_z <= z_range:
            z_min = z_min
            z_max = z_min + z_range * voxel_size_z

            point_cloud_ranges[2] = z_min
            point_cloud_ranges[5] = z_max
            break
        else:
            print('Restarting, becuz condition 1 (z-axis point cloud range/voxel size = 40) is not met')
            print(f'Current voxel_size_z, point_cloud_ranges: {voxel_size_z}, {point_cloud_ranges}')
            print(f'Current value: {(z_max - z_min) / voxel_size_z}')

            # decrease voxel size
            voxel_size_z -= 0.01
            if voxel_size_z <= 0.01: voxel_size_z = 0.01
            voxel_size_z = np.around(voxel_size_z, decimals=2)

            voxel_size_y += 0.01
            if voxel_size_y <= 0.01: voxel_size_y = 0.01
            voxel_size_y = np.around(voxel_size_y, decimals=2)

            voxel_size_x += 0.01
            if voxel_size_x <= 0.01: voxel_size_x = 0.01
            voxel_size_x = np.around(voxel_size_x, decimals=2)

            # decrease magnifying factor
            magnify_factor -= 1
            if magnify_factor <=1: magnify_factor = 1

            # restart point_cloud_ranges
            point_cloud_ranges = [ [], [], [], [], [], [] ]
            
            # remove previous things
            shutil.rmtree(new_directory)

            # recreate the directory
            os.mkdir(new_directory)
            os.mkdir(os.path.join(new_directory, 'ImageSets'))
            os.mkdir(os.path.join(new_directory, 'points'))
            os.mkdir(os.path.join(new_directory, 'labels'))

            continue
    else:
        break
        
print(point_cloud_ranges)

# auto replace point cloud ranges in the dataset yaml file
with open(data_cfg_file, 'r+') as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        if "POINT_CLOUD_RANGE" in line:
            line = f"POINT_CLOUD_RANGE: {str(point_cloud_ranges)}\n"
            lines[i] = line
        if "VOXEL_SIZE" in line:
            line = line.split("VOXEL_SIZE")[0]
            line += f"VOXEL_SIZE: {str([voxel_size_x, voxel_size_y, voxel_size_z])}\n"
            lines[i] = line      
    file.seek(0)
    file.writelines(lines)
    file.truncate()

# auto replace for PointPillar
if model_name in ['PointPillar']:
    with open(cfg_file, 'r+') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # modify POINT_CLOUD_RANGE
            if "POINT_CLOUD_RANGE" in line:
                line = line.split("POINT_CLOUD_RANGE")[0]
                line += f"POINT_CLOUD_RANGE: {str(point_cloud_ranges)}\n"
                lines[i] = line
            # modify VOXEL_SIZE
            if "VOXEL_SIZE" in line:
                line = line.split("VOXEL_SIZE")[0]
                line += f"VOXEL_SIZE: {str([voxel_size_x, voxel_size_y, (np.around(point_cloud_ranges[5] - point_cloud_ranges[2], decimals=2))])}\n"
                lines[i] = line
        file.seek(0)
        file.writelines(lines)
        file.truncate()

