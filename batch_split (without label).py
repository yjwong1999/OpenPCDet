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
parser.add_argument('--dir', type = str, required = True, default="data_raw/preprocessed_techpartnerfile-ply", \
                        help="input folder that contains all the .ply file (after preprocessing)")

args = parser.parse_args()
directory = args.dir

file_names = sorted(os.listdir(directory))

print()
if "part1" in file_names[0] or "part2" in file_names[0]:
    print('Apparently, you have spliited the preprocessed data previously! Please check')
else:
    for idx in tqdm(range(len(file_names)), desc =f'Preprocessing (downsample + filter) input data...'):
        file_name = file_names[idx]
        file_path = os.path.join(directory, file_name)
        save_path_1 = os.path.join(directory, file_name.replace(".ply", "_part1.ply"))
        save_path_2 = os.path.join(directory, file_name.replace(".ply", "_part2.ply"))
        save_paths = [save_path_1, save_path_2]

        if not file_name.endswith('.ply'):
            continue

        pcd = o3d.io.read_point_cloud(file_path)

        inlier_cloud_np = np.array(pcd.points) # the loaded is already inlier clouds, filtered in batch_preprocess.py


        # split into half
        x_range = np.max(inlier_cloud_np[:,0]) + np.min(inlier_cloud_np[:,0])
        thresh = x_range / 2

        part_1 = inlier_cloud_np[np.where(inlier_cloud_np[:,0]>thresh)]
        part_1 = part_1[np.where(part_1[:,0]<0.9*x_range)]

        part_2 = inlier_cloud_np[np.where(inlier_cloud_np[:,0]<=thresh)]
        part_2 = part_2[np.where(part_2[:,0]>0.1*x_range)]

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
