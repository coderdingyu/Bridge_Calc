# 🚀 Bridge Defect Detection - Mask R-CNN 项目

## 📌 项目概览

基于 Mask R-CNN 的桥梁病害自动检测系统，支持 6 类病害的目标检测与实例分割。

## 🎯 核心指标

### 数据集规模
| 项目 | 数值 |
|------|------|
| 病害类别 | 6 种 |
| 训练样本 | 8,454 张 |
| 验证样本 | 1,112 张 |
| 测试样本 | 1,113 张 |
| 总标注数 | 44,702 个 |

### mAP 评估结果 (阈值 0.001)

#### BBox 检测
| 指标 | AP |
|------|-----|
| AP@[0.5:0.95] | 11.9% |
| AP@0.5 | 24.8% |
| AP@0.75 | 10.7% |
| AR@[0.5:0.95] | 36.6% |

#### Segmentation 分割
| 指标 | AP |
|------|-----|
| AP@[0.5:0.95] | 9.4% |
| AP@0.5 | 19.7% |
| AP@0.75 | 7.8% |
| AR@[0.5:0.95] | 28.6% |

### 每类 AP 分析

| 类别 | BBox AP | Segm AP | 问题诊断 |
|------|---------|---------|----------|
| Hole | **23.4%** | **22.0%** | 最佳 |
| ReinForcement | 16.3% | 11.0% | 一般 |
| Comb | 10.5% | 10.3% | 一般 |
| Crack | 8.1% | **0.3%** | ⚠️ Segm 极低 |
| Seepage | 7.0% | 6.5% | 较差 |
| Breakage | 6.4% | 6.2% | 较差 |

**关键发现**: Crack 类的 Segmentation AP 极低 (0.3%)，与其 BBox AP (8.1%) 差距巨大，表明裂缝的 mask 预测存在严重问题。

## � 项目结构

```
Bridge_Calc/
├── train_final.py              训练脚本
├── inference_final.py          推理脚本
├── evaluate_map.py             mAP 评估脚本（含每类 AP）
├── labelme2coco.py            数据格式转换
├── bridge_quantify.py          量化模块
├── auto_pipeline.py            自动流程脚本
├── monitor_training.sh         训练监控
├── models/                     模型权重
│   └── maskrcnn_bridge_best.pth
└── test_results/               推理结果

/data1/dingyu/datasets/labelme/
├── train/val/test/            数据集
└── *_coco.json                COCO 格式标注
```

## � 常用命令

```bash
# 环境激活
conda activate maskrcnn_env
cd /home/dingyu/Bridge_Calc

# 推理
python inference_final.py models/maskrcnn_bridge_best.pth /path/to/image.jpg

# 评估（含每类 AP）
python evaluate_map.py models/maskrcnn_bridge_best.pth

# 量化
python inference_final.py models/maskrcnn_bridge_best.pth test_results/ --enable-quant --save-json --save-csv
```

## ⚠️ 已识别问题

1. **Crack Segmentation 异常**: mask AP 仅 0.3%，需检查 labelme2coco.py 对裂缝的处理
2. **训练轮数不足**: 仅 10 epochs，建议 24-36 epochs
3. **学习率衰减过快**: StepLR(step_size=3) 导致第 3、6、9 epoch 就衰减

## 📝 版本信息

- **最后更新**: 2026-03-30
- **状态**: 训练完成，评估完成，量化待运行
