import logging
import time

import numpy as np
import torch
from detectron2 import utils
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from segmentationsg.data import add_dataset_config, VisualGenomeTrainData, register_datasets
from segmentationsg.modeling.roi_heads.scenegraph_head import add_scenegraph_config
from detectron2.engine import default_argument_parser, default_setup, launch
import cv2
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.config import CfgNode as CN
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


def register_coco_data(args):
    # annotations = json.load(open('/h/skhandel/SceneGraph/data/coco/instances_train2014.json', 'r'))
    # classes = [x['name'] for x in annotations['categories']]
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    MetadataCatalog.get('coco_train_2014').set(thing_classes=classes, evaluator_type='coco')


def setup():
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert (cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']), "Mode {} not supported".format(
        cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file('configs/predictor_sg_dev_masktransfer.yaml', )
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = 'weights/model_0002999.pth'
    cfg.freeze()
    register_coco_data(cfg)
    register_datasets(cfg)
    # default_setup(cfg, args)
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")

    return cfg


# Usage

predictor = DefaultPredictor(setup())


def build_input(image_path):
    dataset_dict = dict()
    dataset_dict["file_name"] = image_path
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    h, w, _ = image.shape
    dataset_dict['width'] = w
    dataset_dict['height'] = h
    utils.check_image_size(dataset_dict, image)

    # if "sem_seg_file_name" in dataset_dict:
    #     sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
    # else:
    sem_seg_gt = None

    aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
    image, sem_seg_gt = aug_input.image, aug_input.sem_seg
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if sem_seg_gt is not None:
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

    dataset_dict["relations"] = torch.as_tensor(np.ascontiguousarray(dataset_dict["relations"]))
    dataset_dict["annotations"] = []
    return dataset_dict


# Example usage:
image_path = 'test_dataset/1.jpg'


start_time = time.time()
with torch.no_grad():
    result = predictor(build_input(image_path))
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(result)
