# 🚀 Bridge Defect Detection System

## 项目概览

基于 Mask R-CNN 的桥梁病害检测系统，可识别 **6 种桥梁结构性缺陷**并进行量化分析。

### 病害类别

| 类别 | 说明 |
|------|------|
| Breakage | 破损 |
| ReinForcement | 钢筋外露 |
| Comb | 蜂窝麻面 |
| Crack | 裂缝 |
| Seepage | 渗水 |
| Hole | 孔洞 |

### 数据规模

| 数据集 | 图像数 | 标注数 |
|--------|--------|--------|
| 训练集 | 8,454 | 34,743 |
| 验证集 | 1,112 | 4,509 |
| 测试集 | 1,113 | 5,450 |
| **合计** | **10,679** | **44,702** |

---

## 快速开始

### 环境

```bash
conda activate maskrcnn_env
```

### 推理

```bash
# 单张图片
python inference_final.py models/maskrcnn_bridge_best.pth /path/to/image.jpg

# 批量推理
python inference_final.py models/maskrcnn_bridge_best.pth /path/to/folder/

# 启用量化
python inference_final.py models/maskrcnn_bridge_best.pth /path/to/folder/ --enable-quant --save-json --save-csv
```

### 评估

```bash
python evaluate_map.py models/maskrcnn_bridge_best.pth
```

---

## 模型性能

### 评估结果 (COCO mAP)

| 指标 | BBox | Segmentation |
|------|------|--------------|
| AP@0.5:0.95 | 11.5% | 9.4% |
| AP@0.5 | 23.1% | 19.7% |
| AR@0.5:0.95 | 36.6% | 28.6% |

### 各类别分布 (测试集)

| 类别 | 数量 |
|------|------|
| Breakage | 3,573 |
| ReinForcement | 1,286 |
| Comb | 216 |
| Crack | 205 |
| Seepage | 113 |
| Hole | 57 |

---

## 项目结构

```
Bridge_Calc/
├── train_final.py              训练脚本
├── inference_final.py          推理脚本
├── evaluate_map.py             mAP评估脚本
├── labelme2coco.py            数据格式转换
├── bridge_quantify.py          量化模块
├── auto_pipeline.py            自动流程脚本
│
├── models/
│   ├── maskrcnn_bridge_best.pth          ⭐ 最佳模型 (169MB)
│   ├── maskrcnn_bridge_epoch_*.pth       各epoch检查点
│
├── test_results/              推理结果 (1113张)
└── TRAINING_MONITORING.md      训练监控文档
```

---

## 依赖

- Python 3.9+
- PyTorch + torchvision
- pycocotools
- OpenCV
- NumPy

---

## 训练配置

- **Epochs**: 10
- **Batch Size**: 2
- **学习率**: 0.005 (StepLR, step=3, gamma=0.1)
- **训练损失**: 0.829825
- **验证损失**: 0.946434

---

## 注意事项

1. 当前模型训练轮数较少（10 epochs），可进一步优化
2. 评估脚本已修复阈值问题（0.5 → 0.001）
3. 所有标注均为polygon格式，无line类型
