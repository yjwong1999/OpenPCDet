import argparse
import glob, os
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
            'filename': self.sample_file_list[index],
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--no-show', default=False, action='store_true', help='Raise this argument if you wish to skip 3D  plotting, only save prediction txt')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # get ready the demo dataset
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    # build and load model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    # prepare directory to store demo output
    output_dir = 'demo_output'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    exp_count = 1
    while True:
        temp = os.path.join(output_dir, f'exp_{exp_count}')
        if not os.path.isdir(temp):
            os.mkdir(temp)
            break
        exp_count += 1
    output_dir = temp
            
    # loop inference
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            
            # get filename
            filename = data_dict['filename']
            filename = os.path.basename(filename)
            filename = os.path.join(output_dir, filename).replace(args.ext, '.txt')
            del data_dict['filename']
            
            # feedforward
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # extract predictions in the format of [x, y, z, dx, dy, dz, rot, cls, conf]
            pred_boxes  = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().detach().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().detach().numpy()
            lines = []
            for i in range(len(pred_labels)):
                pred_box   = pred_boxes[i]
                pred_score = pred_scores[i]
                pred_label = pred_labels[i]
                
                x, y, z, dx, dy, dz, rot = pred_box
                cls = pred_label
                conf = pred_score
                line = [x, y, z, dx, dy, dz, rot, cls, conf]
                line = [str(item) for item in line]
                line = ' '.join(line)
                lines.append(line)
            
            with open(filename, 'w') as file:
                for line in lines:
                    file.write(line + '\n')

            # draw
            if not args.no_show:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
