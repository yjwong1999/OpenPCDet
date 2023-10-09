import open3d as o3d
import numpy as np
import os, json
from math import cos, sin
from tqdm import tqdm
import argparse

'''
# reference
http://www.open3d.org/docs/release/python_example/io/index.html
http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Paint-point-cloud
http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html
'''

################################
# Check if a point in bbox or not
################################
def check_point_within_bbox(point, x_centroid, y_centroid, z_centroid,
                                length, width, height, rotation):

    # Calculate half-dimensions for convenience
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2
    
    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0, 0, 1]])
    
    # Calculate inverse rotation matrix
    inverse_rotation_matrix = np.array([[np.cos(-rotation), -np.sin(-rotation), 0],
                                        [np.sin(-rotation), np.cos(-rotation), 0],
                                        [0, 0, 1]])
    
    # Translate the point relative to the bbox's centroid
    translated_point = point - np.array([x_centroid, y_centroid, z_centroid])
    
    # Apply inverse rotation to the translated point
    aligned_point = np.dot(translated_point, inverse_rotation_matrix.T)
    
    # Check if the aligned point is within the bbox
    is_within_bbox = (
        np.abs(aligned_point[0]) <= half_length and
        np.abs(aligned_point[1]) <= half_width and
        np.abs(aligned_point[2]) <= half_height
    )
    
    #print("Point is within bbox:", is_within_bbox)
    return is_within_bbox

################################
# Augment
################################
def rotate_point(point, rot_angle_z, rot_origin_x, rot_origin_y):
    # extract xyz of point to be rotated
    x = point[0]
    y = point[1]
    z = point[2]
    new_x = np.cos(rot_angle_z) * (x - rot_origin_x) - np.sin(rot_angle_z) * (y - rot_origin_y) + rot_origin_x
    new_y = np.sin(rot_angle_z) * (x - rot_origin_x) + np.cos(rot_angle_z) * (y - rot_origin_y) + rot_origin_y
    new_point = np.array([new_x, new_y, z])
    return new_point
    
def augment(np_pcd, obj, aug_type=None):
    '''
    all augmentation type
    -1: no augment
    0: remove obj
    1: move upward
    2: top/down flip
    3: rotate left/right

    augmentation is performed at two sections (refer AUGMENT SECTION header below)
    '''
    # no augment
    if aug_type == -1:
        return np_pcd, obj
    elif aug_type == 0:
        pass
    elif aug_type == 1:
        displacement = np.random.uniform(0.2, 0.4)
    elif aug_type == 2:
        pass
    elif aug_type == 3:
        rot_angle_z = np.random.uniform(-0.6, 0.6)

    # extract bbox properties
    x_centroid = obj['centroid']['x']
    y_centroid = obj['centroid']['y']
    z_centroid = obj['centroid']['z']
    length = obj['dimensions']['length']
    width = obj['dimensions']['width']
    height = obj['dimensions']['height']
    rotation = obj['rotations']['z']

    selected_points = []
    selected_indexs = []

    ave_x, ave_y = [], [] # to calculate new centroid (for type 3: rotate left/right)
    for i in range(len(np_pcd)):
        point = np_pcd[i]
        is_within_bbox = check_point_within_bbox(point, x_centroid, y_centroid, z_centroid,
                                                    length, width, height, rotation)
        # if the point is in the bbox
        if is_within_bbox:
            selected_points.append(point)
            selected_indexs.append(i)
            
            ######################
            # AUGMENT SECTION 1
            ######################
            # 0: if delete
            if aug_type == 0: 
                # no augmentation required here, just have to delete the points in Section 2
                pass
                
            # 1: if move upward
            elif aug_type == 1: 
                new_point = point + np.array([0, 0, 0.2])
                np_pcd[i] = new_point    
            
            # 2: if top/down flip
            elif aug_type == 2:
                # no augmentation required here, do in Section 2
                pass
            
            # 3: rotate left/right (< y mid)
            # find the origin for rotation
            elif aug_type == 3:
                if y_centroid < y_mid:
                    # dunno how to explain but yup, we need two part
                    rot_origin_x = x_centroid
                    rot_origin_y = (y_centroid + width / 2)
                    origin_1 = rotate_point([rot_origin_x, rot_origin_y, -1], -1 * (3.14159265 - rotation), x_centroid, y_centroid)
                    x1, y1 = origin_1[0], origin_1[1]

                    # dunno how to explain but yup, we need two part
                    rot_origin_x = x_centroid
                    rot_origin_y = (y_centroid - width / 2)
                    origin_2 = rotate_point([rot_origin_x, rot_origin_y, -1], -1 * (3.14159265 - rotation), x_centroid, y_centroid)
                    x2, y2 = origin_2[0], origin_2[1]

                    if y1 < y2:
                        rot_origin_x = x1
                        rot_origin_y = y1
                    else:
                        rot_origin_x = x2
                        rot_origin_y = y2

                else:
                    # dunno how to explain but yup, we need two part
                    rot_origin_x = x_centroid
                    rot_origin_y = (y_centroid + width / 2)
                    origin_1 = rotate_point([rot_origin_x, rot_origin_y, -1], -1 * (3.14159265 - rotation), x_centroid, y_centroid)
                    x1, y1 = origin_1[0], origin_1[1]

                    # dunno how to explain but yup, we need two part
                    rot_origin_x = x_centroid
                    rot_origin_y = (y_centroid - width / 2)
                    origin_2 = rotate_point([rot_origin_x, rot_origin_y, -1], -1 * (3.14159265 - rotation), x_centroid, y_centroid)
                    x2, y2 = origin_2[0], origin_2[1]

                    if y1 > y2:
                        rot_origin_x = x1
                        rot_origin_y = y1
                    else:
                        rot_origin_x = x2
                        rot_origin_y = y2

                # rotation angle at z            
                rot_angle_z = 0.6 #np.random.uniform(0.5, 0.7, 1)[0] # radian
                #if np.random.rand() > 0.5:
                    #rot_angle_z *= -1

                # edit the point
                new_point = rotate_point(point, rot_angle_z, rot_origin_x, rot_origin_y)
                np_pcd[i] = new_point
                
                # update ave_x, ave_y to calculate new centroid
                ave_x.append(new_point[0])
                ave_y.append(new_point[1])


    ######################
    # AUGMENT SECTION 2
    ###################### 
    # 0: if delete
    if aug_type == 0:
        # augment ply
        np_pcd = np.delete(np_pcd, selected_indexs, axis=0)
        
        # augment label
        obj = None

    # 1: if move upward    
    elif aug_type == 1:   
        # augment label
        obj['centroid']['z'] = obj['centroid']['z'] + 0.2

    # 2: if top/down flip
    elif aug_type == 2: 
        # augment ply
        selected_points = np.array(selected_points) 
        z_plane = z_min = np.min(selected_points[:, 2]) 
        z_coords = selected_points[:, 2]
        z_coords = z_coords - z_plane # bring down the object to z=0 plane
        z_coords = -1 * z_coords # flip the opject at z=0
        z_coords = z_coords * 0.5 # squeeze in z dimension
        z_coords = z_coords + z_plane # bring back up
        selected_points[:, 2] = z_coords 
        np_pcd[selected_indexs] = selected_points
        
        # augmetn label similarly
        obj['centroid']['z'] = (-1 * (z_centroid - z_plane)) * 0.5 + z_plane
        obj['dimensions']['height'] = obj['dimensions']['height'] * 0.5
    
    # 3: rotate left/right
    elif aug_type == 3:
        # augment label
        x_centroid = sum(ave_x) / len(ave_x)
        y_centroid = sum(ave_y) / len(ave_y)
        rotation = rotation + rot_angle_z
        
        obj['centroid']['x'] = x_centroid
        obj['centroid']['y'] = y_centroid
        obj['rotations']['z'] = rotation      

    if obj is not None: obj['name'] = 'defect'

    return np_pcd, obj

################################
# Loop ply files
################################

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--label_dir', type = str, required = False, default = 'custom', help="where is the directory for the labels")
parser.add_argument('--ply_dir', type = str, required = False, default = 'custom', help="where is the directory for the ply data")
args = parser.parse_args()

# start looping the data
print("Load a ply point cloud, then augment it")

ply_dir = os.path.join(os.getcwd(), args.ply_dir)
label_dir = os.path.join(os.getcwd(), args.label_dir)

filenames = os.listdir(label_dir)
for fn_idx in tqdm(range(len(filenames)), desc =f'Data Augmentation'):
    # augment 2 times for each ply
    for aug_idx in range(2):
    
        filename = filenames[fn_idx]
        ################################
        # read ply
        ################################
        ply_filename = os.path.join(ply_dir, filename).replace('.json', '.ply')
        pcd = o3d.io.read_point_cloud(ply_filename)
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(np.array(pcd.points)) 
        #o3d.visualization.draw_geometries([temp])
        np_pcd = np.asarray(pcd.points)
        y_mid = (np.max(np_pcd[:,1]) + np.min(np_pcd[:,1])) / 2

        ################################
        # read label
        ################################
        label_filename = os.path.join(label_dir, filename)
        with open(label_filename) as f:
            d = json.load(f)
            objs = d['objects']

        if len(objs) != 13:
            continue

        ################################
        # arrange the obj following the sequence
        ################################
        all_x_c, all_y_c = [], []
        for obj in objs:
            x_centroid = obj['centroid']['x']
            y_centroid = obj['centroid']['y']
            all_x_c.append(x_centroid)
            all_y_c.append(y_centroid)

        # get y mid
        y_mid = (max(all_y_c) + min(all_y_c)) / 2

        # arrange according to x value, in ascending order
        all_x_c = np.array(all_x_c)  
        new_objs = np.array(objs)
        idx = np.argsort(all_x_c) 
        new_objs = list(new_objs[idx])

        # split to bottom and top
        new_objs_1, new_objs_2 = [], []
        for obj in new_objs:
            y_centroid = obj['centroid']['y']
            if y_centroid <= y_mid:
                new_objs_1.append(obj)
            else:
                new_objs_2.append(obj)

        # combine part bottom and top
        objs = new_objs_1 + new_objs_2
     

        ################################
        # create augmentation config
        ################################
        aug_list = [-1 for i in range(len(objs))]
        list_of_objs_yet_to_augment = [i for i in range(len(objs))]
        np.random.shuffle(list_of_objs_yet_to_augment)

        # random rotate a few legs
        if np.random.rand() > 0.5:
            aug_list[0] = 3
            list_of_objs_yet_to_augment.remove(0)
        if np.random.rand() > 0.5:
            aug_list[3] = 3
            list_of_objs_yet_to_augment.remove(3)

        # random remove 4 legs
        for i in range(4):
            idx = list_of_objs_yet_to_augment.pop(0)
            aug_list[idx] = 0

        # at least 5 legs are normal
        for i in range(5):
            idx = list_of_objs_yet_to_augment.pop(0)
            aug_list[idx] = -1

        # random augment the rest using aug_type = 1 or 2
        for i in list_of_objs_yet_to_augment:
            aug_list[i] = np.random.choice([1, 2])

        print(aug_list)

        ################################
        # loop all legs for augmentation
        ################################
        #aug_list = [-1 for i in range(len(objs))]
        #aug_list[3] = 3
        final_objs = []
        for obj_idx, obj in enumerate(objs):
            #print(obj)

            # augment here
            rand_int = aug_list[obj_idx] #np.random.randint(0, high=3+1, size=None, dtype=int)
            np_pcd, obj = augment(np_pcd, obj, aug_type=rand_int)
            if obj is not None:
                final_objs.append(obj)

        # plot the augmented points
        augmented_points = np.array(np_pcd)
        color_grad = o3d.geometry.PointCloud()
        color_grad.points = o3d.utility.Vector3dVector(augmented_points) 
        #o3d.visualization.draw_geometries([color_grad])
        
        # save ply and label
        new_ply_filename = os.path.join(ply_dir, f'augmented_00{aug_idx} ' + filename).replace('.json', '.ply')
        o3d.io.write_point_cloud(new_ply_filename, color_grad) # save the filtered point cloud
        
        new_label_filename = os.path.join(label_dir, f'augmented_00{aug_idx} ' + filename)
        d['folder'] = os.path.join(os.getcwd(), ply_dir)
        d['filename'] = os.path.basename(new_ply_filename)
        d['path'] = new_ply_filename
        d['objects'] = final_objs
        with open(new_label_filename, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)



