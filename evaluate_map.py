#!/usr/bin/env python3
"""
Mask R-CNN 终极评估脚本 (COCO mAP 标准)
用于计算测试集上的严格定量指标 (mAP for BBox & Segmentation)
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import warnings

warnings.filterwarnings('ignore')

NUM_CLASSES = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_JSON_PATH = "/data1/dingyu/datasets/labelme/test_coco.json"
TEST_IMG_DIR = "/data1/dingyu/datasets/labelme/test"

def load_model(model_path):
    print(f"\n📦 加载评估模型: {model_path}")
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, NUM_CLASSES)
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def main():
    print("\n" + "=" * 70)
    print("  🏆 Mask R-CNN 终极大考评估 (mAP 计算)")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\n❌ 缺少模型路径！用法: python evaluate_map.py <model.pth>\n")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if not os.path.exists(model_path) or not os.path.exists(TEST_JSON_PATH):
        print(f"❌ 模型或测试集文件不存在")
        sys.exit(1)

    model = load_model(model_path)
    
    print("\n📊 正在加载测试集标准答案 (Ground Truth)...")
    coco_gt = COCO(TEST_JSON_PATH)
    image_ids = coco_gt.getImgIds()
    
    coco_results = []
    print("\n🚀 开始执行全量盲测推理 (这可能需要几分钟)...")
    for img_id in tqdm(image_ids, desc="评估进度", ncols=100):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(TEST_IMG_DIR, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        image_pil = Image.open(img_path).convert('RGB')
        image_tensor = torch.from_numpy(np.array(image_pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(image_tensor)[0]
            
        boxes, scores, labels, masks = predictions['boxes'].cpu().numpy(), predictions['scores'].cpu().numpy(), predictions['labels'].cpu().numpy(), predictions['masks'].cpu().numpy()
        
        for i in range(len(scores)):
            box = boxes[i]
            label = labels[i]
            score = scores[i]
            mask = masks[i]
            
            if score < 0.5:
                continue
            
            # Convert box format from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
            x_min, y_min, x_max, y_max = box
            width = max(0, x_max - x_min)
            height = max(0, y_max - y_min)
            
            # Encode mask to RLE
            mask_binary = (mask > 0.5).astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask_binary[0]))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Create COCO result
            result = {
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [float(x_min), float(y_min), float(width), float(height)],
                'score': float(score),
                'segmentation': rle
            }
            coco_results.append(result)

    if not coco_results:
        print("⚠️ 未生成任何检测结果")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("  📈 执行官方 COCO 评估算法...")
    print("=" * 70)
    
    coco_dt = coco_gt.loadRes(coco_results)
    
    print("\n" + "-" * 20 + " 1. 边界框 (BBox) 检测精度 " + "-" * 20)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    print("\n" + "-" * 20 + " 2. 多边形掩码 (Segmentation) 抠图精度 " + "-" * 20)
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()

if __name__ == "__main__":
    main()
