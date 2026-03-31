#!/usr/bin/env python3
"""创建一个虚拟模型用于测试"""
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 7

model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, NUM_CLASSES)

torch.save(model.state_dict(), "/home/dingyu/Bridge_Calc/dummy_model.pth")
print("虚拟模型已创建: /home/dingyu/Bridge_Calc/dummy_model.pth")
