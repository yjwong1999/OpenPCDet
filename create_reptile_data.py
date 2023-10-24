import os, shutil, copy
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val-num', type = int, required = True, default = 1, help="how many samples from each fail type as validation")
parser.add_argument('--upsample', type = int, required = True, default = 10, help="upsample defect files")
args = parser.parse_args()

# remove previous model
try:
    shutil.rmtree('/output/custom_models/pointpillar')
except:
    pass
'''
# read validation data
val_txt = 'data/custom/ImageSets/val.txt'
val_data = []
with open(val_txt) as f:
    for line in f:
        val_data.append(line.replace('\n', ''))
        #print(line.replace('\n', ''))
'''
# split filename based on defect type
LEG_COUNTS = 13
Z_THRESH_LOW = 13
Z_THRESH_HIGH = 22

custom_data_dir = 'data/custom/labels'
label_filenames = os.listdir(custom_data_dir)
pass_filenames = []
fail_1_filenames = []
fail_2_filenames = []
fail_3_filenames = []
fail_4_filenames = []
for filename in label_filenames:
    if True or 'augmented' not in filename:
        filename = os.path.join(custom_data_dir, filename)

        with_down = False
        with_up = False
        with_other = False
        with_missing = False
        count = 0
        with open(filename) as f:
            for line in f:
                count+=1
                if 'fail' in line:
                    z = float(line.split()[2])
                    if z >= Z_THRESH_HIGH:
                        with_up = True
                    elif z <= Z_THRESH_LOW:
                        with_down = True
                    else:
                        with_other = True
        with_missing = count != LEG_COUNTS

        if with_down:
            fail_1_filenames.append(filename)
        elif with_up:
            fail_2_filenames.append(filename)
        elif with_other:
            fail_3_filenames.append(filename)
        elif with_missing:
            fail_4_filenames.append(filename)
        else:
            pass_filenames.append(filename)

all_fail_filenames = [fail_1_filenames, fail_2_filenames, fail_3_filenames, fail_4_filenames]
all_pass_filenames = copy.deepcopy(pass_filenames)

# take VAL_N sample from each type of fails for validation (1 in this case becuz too less)
VAL_N = args.val_num
UPSAMPLE = args.upsample
NUM_TASK = len(all_fail_filenames)

val_filenames = []
for task_idx in range(NUM_TASK):
    fail_filenames = all_fail_filenames[task_idx]
    for _ in range(VAL_N):
        try:
            val_filenames.append(fail_filenames.pop(0))
        except:
            print(f'fail {task_idx}') #raise KeyboardInterrupt

# take 20% of pass filenames as validation
print(len(val_filenames))
val_filenames += all_pass_filenames[:int(0.2*len(all_pass_filenames))]
all_pass_filenames = all_pass_filenames[int(0.2*len(all_pass_filenames)):]
print(len(val_filenames))

# remove original ImageSets
try:
    os.remove('data/custom/ImageSets/train.txt')
    os.remove('data/custom/ImageSets/val.txt')
except:
    pass 

# group the filenames into tasks
chunck_len = int(len(all_pass_filenames) * (1 / NUM_TASK))
for task_idx in range(NUM_TASK):
    # i/j is start/end index
    i = task_idx * chunck_len
    j = i + chunck_len
    if task_idx == NUM_TASK - 1:
        j = len(all_pass_filenames)

    # pass filename
    pass_filenames = all_pass_filenames[i:j]

    # fail filename
    fail_filenames = all_fail_filenames[task_idx]
    print(f'Task {task_idx+1}: {len(pass_filenames)} pass files, {len(fail_filenames)} fail files (before upsample)')

    # train filename
    with open(f'data/custom/ImageSets/reptile_train_{task_idx+1}.txt', 'w') as f:
        train_filenames = pass_filenames + fail_filenames * UPSAMPLE
        np.random.seed(0)
        np.random.shuffle(train_filenames)
        for fname in train_filenames:
            fname = os.path.splitext(os.path.basename(fname))[0] + '\n'
            f.write(fname)


# validation filename
with open('data/custom/ImageSets/val.txt', 'w') as f:
    train_filenames = pass_filenames + fail_filenames
    for fname in train_filenames:
        fname = os.path.splitext(os.path.basename(fname))[0] + '\n'
        f.write(fname)
