import open3d as o3d
import numpy as np
import os, shutil
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
parser.add_argument('--input-dir', type = str, required = True, default="data_raw/techpartnerfile-ply/techpartnerfile-ply", \
                        help="input folder that contains all the .ply file (before preprocessing)")
parser.add_argument('--output-dir', type = str, required = True, default="data_raw/preprocessed_techpartnerfile-ply", \
                        help="input folder that contains all the .ply file (after preprocessing)")

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

file_names = os.listdir(input_dir)

# if output directory doesn't exist
if not os.path.isdir(output_dir):
    # Create the output directory
    os.makedirs(output_dir, exist_ok=False)

    for idx in tqdm(range(len(file_names)), desc =f'Preprocessing (downsample + filter) input data...'):
        file_name = file_names[idx]
        file_path = os.path.join(input_dir, file_name)
        save_path = os.path.join(output_dir, file_name)

        if not file_name.endswith('.ply'):
            continue

        pcd = o3d.io.read_point_cloud(file_path)

        uni_down_pcd = pcd.uniform_down_sample(every_k_points=8)
        cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)

        inlier_cloud = uni_down_pcd.select_by_index(ind) # filter out inliers

        inlier_cloud_np = np.array(inlier_cloud.points)
        filtered_pcd = o3d.geometry.PointCloud() # create a point cloud object to store the filtered points
        filtered_pcd.points = o3d.utility.Vector3dVector(inlier_cloud_np) # Pass xyz to Open3D.o3d.geometry.PointCloud

        o3d.io.write_point_cloud(save_path, filtered_pcd) # save the filtered point cloud
else:
    print('Apparently, you have preprocessed the input data previously! Please check')
