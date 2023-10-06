import open3d as o3d
import numpy as np
import os, shutil, json, copy
import argparse
from tqdm import tqdm

'''
# reference
http://www.open3d.org/docs/release/python_example/io/index.html
http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Paint-point-cloud
http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html
'''

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--ply_dir', type = str, required = True, default="data_raw/techpartnerfile/preprocessed_techpartnerfile-ply", \
                        help="input folder that contains all the .ply file (after preprocessing)")
parser.add_argument('--label_dir', type = str, required = True, default="data_raw/techpartnerfile/techpartnerfile_label", \
                        help="input folder that contains all labels (json file)")

args = parser.parse_args()
ply_dir = args.ply_dir
label_dir = args.label_dir

file_names = sorted(os.listdir(ply_dir))

print()
if "part1" in file_names[0] or "part2" in file_names[0]:
    print('Apparently, you have spliited the preprocessed data previously! Please check')
else:
    for idx in tqdm(range(len(file_names)), desc =f'Splitting and cropping the RoRI of input data...'):
        ####################################
        # Split data (ply)
        ####################################
        file_name = file_names[idx]
        if not file_name.endswith('.ply'):
            continue
        file_path = os.path.join(ply_dir, file_name)
        save_path_1 = os.path.join(ply_dir, file_name.replace(".ply", "_part1.ply"))
        save_path_2 = os.path.join(ply_dir, file_name.replace(".ply", "_part2.ply"))
        save_paths = [save_path_1, save_path_2]

        pcd = o3d.io.read_point_cloud(file_path)

        inlier_cloud_np = np.array(pcd.points) # the loaded is already inlier clouds, filtered in batch_preprocess.py


        # split into half
        x_range = 5.13 # the ROI is fixed, dont use np.max(inlier_cloud_np[:,0]) + np.min(inlier_cloud_np[:,0])
        x_mid_thresh = x_range / 2
        x_min_thresh = 0.23
        x_max_thresh = 0.77

        part_1 = inlier_cloud_np[np.where(inlier_cloud_np[:,0]>x_mid_thresh)]
        part_1 = part_1[np.where(part_1[:,0]<x_max_thresh*x_range)]
        part_1[:,0] = part_1[:,0] - x_mid_thresh # update the xyz value in point cloud itself

        part_2 = inlier_cloud_np[np.where(inlier_cloud_np[:,0]<=x_mid_thresh)]
        part_2 = part_2[np.where(part_2[:,0]>x_min_thresh*x_range)]
        part_2[:,0] = part_2[:,0] - x_min_thresh*x_range  # update the xyz value in point cloud itself

        inlier_cloud_nps = [part_1, part_2]

        # save
        for inlier_cloud_np, save_path in zip(inlier_cloud_nps, save_paths):
            splitted_pcd = o3d.geometry.PointCloud() # create a point cloud object to store the filtered points
            splitted_pcd.points = o3d.utility.Vector3dVector(inlier_cloud_np) # Pass xyz to Open3D.o3d.geometry.PointCloud

            o3d.io.write_point_cloud(save_path, splitted_pcd) # save the filtered point cloud

            #o3d.visualization.draw_geometries([splitted_pcd])
        #o3d.visualization.draw_geometries([pcd])

        # remove ori (non-spliited ply file)
        os.remove(file_path)

        ####################################
        # Split label (json) if available
        ####################################
        try:
            file_name = file_names[idx]
            ori_label_path = os.path.join(label_dir, file_name.replace(".ply", ".json"))
            label_path_1 = os.path.join(label_dir, file_name.replace(".ply", "_part1.json"))
            label_path_2 = os.path.join(label_dir, file_name.replace(".ply", "_part2.json"))
            label_paths = [label_path_1, label_path_2]

            # read ori label json
            with open(ori_label_path) as f:
                data = json.load(f)

            # modify the label accordingly
            objs = data['objects']
            objs_part_1 = []
            objs_part_2 = []
            for obj in objs:
                if obj['centroid']['x'] > x_mid_thresh:
                    obj['centroid']['x'] = obj['centroid']['x'] - x_mid_thresh
                    objs_part_1.append(obj)
                else:
                    obj['centroid']['x'] = obj['centroid']['x'] - x_min_thresh*x_range
                    objs_part_2.append(obj)
            splitted_objs = [objs_part_1, objs_part_2]

            # save
            for i, (label_path, objs) in enumerate(zip(label_paths, splitted_objs)):
                new_data = copy.deepcopy(data)
                new_data['objects'] = objs
                new_data['filename'] = new_data['filename'].replace(".ply", f"_part{i+1}.ply")
                new_data['path'] = new_data['path'].replace(".ply", f"_part{i+1}.ply")

                with open(label_path, 'w') as f:
                    json.dump(new_data, f, indent=4)

            # remove ori
            os.remove(ori_label_path)
        except Exception as e:
            print(e)
