#!/usr/bin/env python3
"""
Mask R-CNN 最终版训练脚本 (与终极防弹版 COCO 格式对齐)
完全兼容修复后的越界坐标数据
"""

import os
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
import sys

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
DATA_DIR = "/data1/dingyu/datasets/labelme"
NUM_CLASSES = 7  # 6 种病害 + 1 个背景
BATCH_SIZE = 2   
EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "/home/dingyu/Bridge_Calc/models"
NUM_WORKERS = 2
# ============================================

os.makedirs(SAVE_DIR, exist_ok=True)

class COCODataset(torch.utils.data.Dataset):
    """完全兼容新格式的 COCO 数据集加载器"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        json_file = os.path.join(data_dir, f"{split}_coco.json")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # 构建 image_id 到 annotations 的映射
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.data_dir, self.split, img_info['file_name'])
        
        # 加载图像
        if not os.path.exists(img_path):
            return None, None
        
        image = Image.open(img_path).convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # 获取该图像的所有标注
        anns = self.img_id_to_anns.get(img_id, [])
        
        masks = []
        boxes = []
        labels = []
        
        for ann in anns:
            # 从分割多边形创建 mask
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            mask = self._segmentation_to_mask(
                ann['segmentation'],
                img_info['height'],
                img_info['width']
            )
            
            if mask.sum() == 0:  # 跳过空 mask
                continue
            
            masks.append(mask)
            
            # bbox 格式：[x_min, y_min, width, height] -> [x_min, y_min, x_max, y_max]
            bbox = ann['bbox']
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            boxes.append([bbox[0], bbox[1], x_max, y_max])
            labels.append(ann['category_id'])
        
        # 处理没有标注的情况
        if len(masks) == 0:
            masks = np.zeros((1, img_info['height'], img_info['width']), dtype=np.uint8)
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        # 转换为 Tensor
        image = torch.from_numpy(image_array).permute(2, 0, 1)
        masks = torch.as_tensor(np.array(masks, dtype=np.uint8))
        boxes = torch.as_tensor(np.array(boxes, dtype=np.float32))
        labels = torch.as_tensor(np.array(labels, dtype=np.int64))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id], dtype=torch.int64)
        }
        
        return image, target
    
    @staticmethod
    def _segmentation_to_mask(segmentation, height, width):
        """将 COCO 分割多边形转换为二值 mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if isinstance(segmentation, list):
            for seg in segmentation:
                if len(seg) < 6:  # 至少 3 个点
                    continue
                points = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], 1)
        
        return mask


def collate_fn(batch):
    """自定义批次整理函数"""
    batch = [x for x in batch if x[0] is not None]
    if len(batch) == 0:
        return None, None
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    print("\n" + "=" * 70)
    print("  🚀 Mask R-CNN 最终版训练 (与终极防弹版 COCO 格式对齐)")
    print("=" * 70)
    print(f"  设备: {DEVICE}")
    print(f"  类别数: {NUM_CLASSES}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练 Epochs: {EPOCHS}\n")
    
    # 加载模型
    print("📦 加载预训练模型...")
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # 修改分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    # 修改 mask 预测器
    mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, NUM_CLASSES)
    
    model = model.to(DEVICE)
    
    # 加载数据集
    print("📂 加载数据集...")
    train_dataset = COCODataset(DATA_DIR, split='train')
    val_dataset = COCODataset(DATA_DIR, split='val')
    
    print(f"   ├─ 训练集 (Train): {len(train_dataset)} 张图像")
    print(f"   └─ 验证集 (Val):   {len(val_dataset)} 张图像\n")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    print("=" * 70)
    print("🔥 开始训练...")
    print("=" * 70 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", ncols=80)
        
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (images, targets) in pbar:
            if images is None:
                continue
            
            # 移到设备
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += losses.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'avg_loss': f'{epoch_loss / batch_count:.4f}'
            })
        
        # 计算训练集平均 Loss
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        
        # ======================== 验证循环 ========================
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        
        val_loss = 0.0
        val_batch_count = 0
        
        pbar_val = tqdm(enumerate(val_loader), total=len(val_loader),
                       desc=f"Epoch [{epoch+1}/{EPOCHS}] Val", ncols=80)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in pbar_val:
                if images is None:
                    continue
                
                # 移到设备
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # 前向传播
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
                val_batch_count += 1
                
                pbar_val.set_postfix({
                    'val_loss': f'{losses.item():.4f}',
                    'avg_val_loss': f'{val_loss / val_batch_count:.4f}'
                })
        
        # 计算验证集平均 Loss
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        # ================== 验证循环结束 ==================
        
        print(f"\n📊 Epoch {epoch+1} 总结:")
        print(f"   ├─ Train Loss: {avg_train_loss:.6f}")
        print(f"   └─ Val Loss:   {avg_val_loss:.6f}")
        
        # 每个 epoch 保存模型
        model_path = os.path.join(SAVE_DIR, f"maskrcnn_bridge_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        
        # 学习率更新
        lr_scheduler.step()
        
        # 记录最佳 Loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(SAVE_DIR, "maskrcnn_bridge_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✨ 最佳模型已保存: {best_model_path}\n")
    
    print("\n" + "=" * 70)
    print("✨ 训练完成!!! 🎉")
    print("=" * 70)
    print(f"\n📌 模型位置:")
    print(f"   ├─ 最终模型: {SAVE_DIR}/maskrcnn_bridge_epoch_{EPOCHS}.pth")
    print(f"   └─ 最佳模型: {SAVE_DIR}/maskrcnn_bridge_best.pth")
    print(f"\n💪 你的团队专属 AI 模型已就绪!")
    print(f"\n📖 下一步：推理验证")
    print(f"   python inference.py models/maskrcnn_bridge_best.pth test_image.jpg\n")


if __name__ == "__main__":
    main()
