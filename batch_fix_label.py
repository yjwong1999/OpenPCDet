# read all json file in a folder

import json
import os
import argparse

# write a json file
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_all_json_files(label_dir, ply_dir):
    file_list = os.listdir(label_dir)
    file_list.sort()
    data_list = []
    for file in file_list:
        if file == '_classes.json':
            print('skip _classes.json')
            continue
        file_path = os.path.join(label_dir, file)
        data = read_json_file(file_path)

        # modify the folder that contains the ply file
        data['folder'] = ply_dir

        # modify the label path
        path = data['path']
        if len(data['objects']) == 0:
            print(f"Remove empty file {file_path}")
            os.remove(file_path)
            continue
        # get the filename without parents directory
        path = path.split('/')[-1]
        path = os.path.join(os.getcwd(), ply_dir, path)
        data['path'] = path
        # write to json file using write_json_file
        write_json_file(file_path, data)
        
# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--label_dir', type = str, required = False, default = 'custom', help="where is the directory for the labels")
parser.add_argument('--ply_dir', type = str, required = False, default = 'custom', help="where is the directory for the ply data")
args = parser.parse_args()


read_all_json_files(args.label_dir, args.ply_dir)
