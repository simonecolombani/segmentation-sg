import logging
import time

import torch
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
    annotations = args.DATASETS.MSCOCO.ANNOTATIONS
    dataroot = args.DATASETS.MSCOCO.DATAROOT
    register_coco_instances("coco_train_2017", {}, annotations + 'instances_train2017.json', dataroot + '/train2017/')
    register_coco_instances("coco_val_2017", {}, annotations + 'instances_val2017.json', dataroot + '/val2017/')


def setup():
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert (cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']), "Mode {} not supported".format(
        cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file('configs/predictor_sg_dev_masktransfer.yaml', )
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = 'weights/model_0014999.pth'
    cfg.freeze()
    register_coco_data(cfg)
    register_datasets(cfg)
    # default_setup(cfg, args)
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")

    return cfg


# Usage

predictor = DefaultPredictor(setup())


def prepare_image(image_path, target_height, target_width):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image
    image = cv2.resize(image, (target_width, target_height))

    # Convert the image to RGB (Detectron2 requires RGB format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image to [0, 1]
    image = image / 255.0

    return image


# Example usage:
image_path = 'test_dataset/3.png'
target_height = 800  # Adjust to your model's input size
target_width = 1200  # Adjust to your model's input size

# Prepare the image
prepared_image = prepare_image(image_path, target_height, target_width)
start_time = time.time()
with torch.no_grad():
    result = predictor(prepared_image)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(result)
