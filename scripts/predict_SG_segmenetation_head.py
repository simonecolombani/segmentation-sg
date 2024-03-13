import logging
import time

import numpy as np
import torch
from detectron2 import utils
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.structures import Instances, PolygonMasks, Boxes
from segmentationsg.data import add_dataset_config, VisualGenomeTrainData, register_datasets
from segmentationsg.modeling.roi_heads.scenegraph_head import add_scenegraph_config
from detectron2.engine import default_argument_parser, default_setup, launch
import cv2
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.config import CfgNode as CN
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import torch
from PIL import Image
import torchvision.transforms as transforms


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
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = 'weights/model_0002999.pth'
    cfg.freeze()
    register_coco_data(cfg)
    register_datasets(cfg)
    # default_setup(cfg, args)
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")

    return cfg


# Usage

predictor = DefaultPredictor(setup())
image_path = 'test_dataset/000000000139.jpg'



# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = utils.read_image(image_path, format="BGR")

# Example usage:


image_values = {
    "file_name": image_path,
    "height": 426,
    "width": 640,
    "image_id": 139,
    "image": img_tensor,
    "instances": Instances(
        image_size=(426, 640),
        #num_instances=20,
        #image_height=800,
        #image_width=1202,
        fields={
            "gt_boxes": Boxes(
                torch.tensor([
                    [445.0781, 267.6244, 491.4677, 398.1408],
                    [13.2032, 315.0422, 293.6449, 493.2019],
                    [1046.5101, 392.8451, 1199.2955, 540.6949],
                    [674.2094, 409.4836, 779.3843, 602.5916],
                    [545.9521, 409.3897, 662.0767, 594.3287],
                    [776.0413, 418.7981, 832.7043, 571.5869],
                    [596.1169, 411.7183, 636.6468, 433.4836],
                    [775.2900, 295.9812, 874.9245, 555.1549],
                    [722.0076, 323.3991, 750.4048, 390.5164],
                    [962.0131, 386.3850, 989.6967, 416.3756],
                    [926.1035, 327.3990, 964.2106, 530.7981],
                    [1135.8337, 574.4413, 1162.7661, 660.2817],
                    [1151.7413, 578.8544, 1175.9316, 666.0657],
                    [840.9680, 227.4554, 867.2054, 268.5446],
                    [1031.2032, 581.0892, 1100.0929, 749.4835],
                    [658.7711, 392.1878, 680.1254, 434.5352],
                    [774.2570, 411.3052, 792.3434, 434.8169],
                    [453.0789, 366.1784, 479.7858, 399.2864],
                    [632.5338, 374.6479, 650.8079, 406.0657],
                    [603.2725, 434.2160, 839.0899, 601.2206]
                ])
            ),
            "gt_classes": torch.tensor([58, 62, 62, 56, 56, 56, 56, 0, 0, 68, 72, 73, 73, 74, 75, 75, 56, 75, 75, 60]),
            "gt_masks": PolygonMasks(polygons=[[]])
        }
    )
}


start_time = time.time()
# Read a PIL image

with torch.no_grad():
    result = predictor(img_tensor)
print(result)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(result)
