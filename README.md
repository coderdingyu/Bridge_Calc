# 🚀 Bridge Defect Mask R-CNN - 完整项目指南

## 📌 项目概览

这是一个**桥梁病害自动检测系统**，使用 Mask R-CNN 深度学习模型识别 6 种常见的桥梁结构性缺陷。

## 🎯 核心指标

| 指标 | 值 |
|------|-----|
| 病害类别 | 6 种 (Breakage, ReinForcement, Comb, Crack, Seepage, Hole) |
| 训练样本 | 8,454 张图像 |
| 验证样本 | 1,112 张图像 |
| 测试样本 | 1,113 张图像 |
| 总标注数 | 44,702 个病害区域 |
| 自动修复 | 420 个越界坐标已修剪 |

## 📁 项目结构

```
/home/dingyu/Bridge_Calc/
├── 📜 labelme2coco.py          ✅ 数据转换脚本（终极防弹版）
├── 📊 train_final.py           ✅ 最终训练脚本
├── 🔍 inference_final.py       ✅ 最终推理脚本
├── 📋 WORKFLOW_GUIDE.sh        ℹ️ 工作流指南
├── 📖 README.md                📖 本文件
├── 🗂️ models/                   💾 模型权重输出目录
│   ├── maskrcnn_bridge_epoch_1.pth
│   ├── maskrcnn_bridge_epoch_2.pth
│   ├── ...
│   ├── maskrcnn_bridge_epoch_10.pth
│   └── maskrcnn_bridge_best.pth ⭐
└── 📁 (已弃用的旧脚本)
    ├── train_maskrcnn.py
    └── train_maskrcnn_simple.py

/data1/dingyu/datasets/labelme/
├── train/
│   ├── *.jpg (图像文件)
│   ├── *.json (标注文件)
│   └── train_coco.json ✅
├── val/
│   ├── *.jpg
│   ├── *.json
│   └── val_coco.json ✅
└── test/
    ├── *.jpg
    ├── *.json
    └── test_coco.json ✅
```

## 🔄 完整工作流

### 第 1 步 ✅ - 数据转换

**状态**: ✅ **已完成**

将 Labelme 格式的标注转换为 COCO 格式，并自动修复越界坐标。

```bash
cd /home/dingyu/Bridge_Calc
conda activate maskrcnn_env
python labelme2coco.py
```

**输出**:
```
✅ [train] 集转换完成！(自动修复了 288 个越界坐标)
   - 图片数量: 8454
   - 标注数量: 34743
   - 跳过文件: 0

✅ [val] 集转换完成！(自动修复了 83 个越界坐标)
   - 图片数量: 1112
   - 标注数量: 4509

✅ [test] 集转换完成！(自动修复了 49 个越界坐标)
   - 图片数量: 1113
   - 标注数量: 5450
```

**关键特性**:
- ✓ 无视大小写文件匹配（Linux 友好）
- ✓ 自动修剪越界坐标（防止训练崩溃）
- ✓ 零误差转换，100% 保留原始标注

### 第 2 步 ✅ - 模型训练

**状态**: ✅ **已完成**

已使用修复后的 COCO 数据完成 Mask R-CNN 完整训练。

```bash
cd /home/dingyu/Bridge_Calc
conda activate maskrcnn_env
python train_final.py
```

**配置参数**:
```
模型架构: Mask R-CNN with ResNet50-FPN
类别数: 7 (6种病害 + 背景)
预训练权重: COCO 预训练
批大小: 2
学习率: 0.005
优化器: SGD (momentum=0.9)
学习率调度: StepLR (step_size=3, gamma=0.1)
总 Epochs: 10
```

**训练监控**:

已完成所有 10 个 Epochs 的训练。查看生成的所有模型检查点：
```bash
ls -lh /home/dingyu/Bridge_Calc/models/
```

**预期产出**: ✅
- ✅ 10 个检查点文件 (epoch 1-10) **[已完成]**
- ✅ 1 个最佳模型文件 (maskrcnn_bridge_best.pth) **[已完成]**
- ✅ 完整的训练日志 **[已完成]**

### 第 3 步 🎯 - 模型推理

**状态**: 🎯 **可直接执行**

用训练好的模型对新图像进行推理和可视化。支持**单张图片**或**整个文件夹批量**测试。

```bash
cd /home/dingyu/Bridge_Calc
conda activate maskrcnn_env
python inference_final.py <model.pth> <image.jpg>
```

**使用示例**:

```bash
# 推理单张图片
python inference_final.py models/maskrcnn_bridge_best.pth /data1/dingyu/datasets/labelme/test/12544.jpg

# 批量推理整个文件夹
python inference_final.py models/maskrcnn_bridge_best.pth /data1/dingyu/datasets/labelme/test

# 特定 epoch 的模型
python inference_final.py models/maskrcnn_bridge_epoch_5.pth /path/to/image.jpg
```

**输出**:

生成带彩色标注的结果图像，位于 `/home/dingyu/Bridge_Calc/test_results/` 目录下。

**单张图片输出示例**:
```
🎨 绘制检测结果...
✅ 完成！检测到 12 个缺陷

📌 检测统计:
   • Breakage: 3 个
   • Crack: 5 个
   • Hole: 2 个
   • Seepage: 2 个

💾 结果已保存: /home/dingyu/Bridge_Calc/test_results/12544_detected.jpg
```

**批量推理输出示例**:
```
✨ 批量测试完成！
📂 生成的图像已保存至: /home/dingyu/Bridge_Calc/test_results

📈 测试集全局缺陷统计 (共发现 243 个缺陷):
   • Crack: 98 个
   • Breakage: 65 个
   • Seepage: 43 个
   • Hole: 22 个
   • ReinForcement: 10 个
   • Comb: 5 个
```

**彩色映射**:
```
Breakage (破损)    → 蓝色 (255, 0, 0)
ReinForcement      → 绿色 (0, 255, 0)
Comb (蜂窝)        → 黄色 (0, 255, 255)
Crack (裂缝)       → 洋红 (255, 0, 255)
Seepage (渗水)     → 青色 (255, 255, 0)
Hole (孔洞)        → 橙色 (255, 165, 0)
```

### 第 4 步 📊 - 模型评估

**状态**: 🎯 **可直接执行**

计算模型在测试集上的标准 COCO mAP 指标（包括 BBox 和 Segmentation）。

```bash
cd /home/dingyu/Bridge_Calc
conda activate maskrcnn_env
python evaluate_map.py models/maskrcnn_bridge_best.pth
```

**输出示例**:
```
🏆 Mask R-CNN 终极大考评估 (mAP 计算)
...
-------- 1. 边界框 (BBox) 检测精度 --------
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.742
...

-------- 2. 多边形掩码 (Segmentation) 抠图精度 --------
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.698
...
```

## 🚀 完整工作流速查

### labelme2coco.py

**配置参数**:
```python
DATA_DIR = "/data1/dingyu/datasets/labelme"  # 数据根目录
SPLITS = ["train", "val", "test"]             # 处理的子集
CATEGORY_MAPPING = {}                         # 自动生成类别映射
```

**核心函数**:
- `get_image_file()`: 无视大小写查找图片文件
- `convert_to_coco()`: 执行格式转换和坐标修复

### train_final.py

**配置参数**:
```python
DATA_DIR = "/data1/dingyu/datasets/labelme"
NUM_CLASSES = 7
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.005
DEVICE = "cuda"
SAVE_DIR = "/home/dingyu/Bridge_Calc/models"
```

**核心类**:
- `COCODataset`: 数据集加载器
- `collate_fn`: 批次整理函数

**输出**:
- 每个 epoch 保存检查点
- 每个 epoch 打印 Loss 指标
- 自动选择最佳模型

### inference_final.py

**配置参数**:
```python
NUM_CLASSES = 7
CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
CLASS_NAMES = {...}         # 类别名称
CLASS_COLORS = {...}        # 可视化颜色
```

**命令行参数**:
```
python inference_final.py <model_path> <image_path>
```

**输出**:
- 检测数量统计
- 各类别分布统计
- 带标注的可视化图像

## 🔧 故障排查

### ❌ 问题: 训练卡住/进度条不动

**原因**: GPU 显存不足或数据加载器堵塞

**解决**:
```bash
# 查看 GPU 状态
nvidia-smi

# 停止训练
pkill -f train_final.py

# 重新启动（可选：减小 BATCH_SIZE）
python train_final.py
```

### ❌ 问题: COCO JSON 不存在

**原因**: `labelme2coco.py` 未运行或运行失败

**解决**:
```bash
# 重新运行转换
python labelme2coco.py

# 验证输出
ls -lh /data1/dingyu/datasets/labelme/*_coco.json
```

### ❌ 问题: 推理结果全黑/无检测

**原因**: 模型文件损坏、路径错误或模型未收敛

**解决**:
1. 检查模型文件是否存在且完整
2. 使用 `maskrcnn_bridge_best.pth` 而不是中间 checkpoint
3. 等待训练完成（至少 5 个 epochs）

### ❌ 问题: 内存不足错误

**原因**: 数据加载器或 GPU 显存不足

**解决**:
```python
# 编辑 train_final.py，调整配置
BATCH_SIZE = 1          # 从 2 改为 1
NUM_WORKERS = 0         # 从 2 改为 0（禁用多进程加载）
```

## 📊 性能指标

| 指标 | 当前值 |
|------|--------|
| 训练样本数 | 8,454 |
| 验证样本数 | 1,112 |
| 平均标注数/图像 | ~4.1 |
| 最大图像分辨率 | 4608x3456 |
| 最小图像分辨率 | 640x480 |
| 预期模型大小 | ~170MB |
| 推理速度/图 | ~1-2 秒 |

## 🚀 快速开始命令

**当前状态（训练已完成）**:

```bash
# 1. 环境激活
conda activate maskrcnn_env
cd /home/dingyu/Bridge_Calc

# 2. 单张推理 (最佳模型)
python inference_final.py models/maskrcnn_bridge_best.pth /path/to/image.jpg

# 3. 批量推理 (测试集)
python inference_final.py models/maskrcnn_bridge_best.pth /data1/dingyu/datasets/labelme/test

# 4. 模型评估 (COCO mAP)
python evaluate_map.py models/maskrcnn_bridge_best.pth

# 5. 查看推理结果
ls -lh /home/dingyu/Bridge_Calc/test_results/
```

## 📞 技术支持

遇到问题时检查：
1. 环境激活: `conda activate maskrcnn_env`
2. 数据完整性: `ls -lh /data1/dingyu/datasets/labelme/*_coco.json`
3. GPU 可用: `nvidia-smi`
4. 日志输出: `tail -50 training_final_v2.log`

## 📝 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-03-27 | 初始版本 |
| 2.0 (最终版) | 2026-03-27 | 终极防弹版（新增坐标修复） |
| 3.0 (训练完成) | 2026-03-27 | 完整训练+验证+评估版本（新增 evaluate_map.py） |

---

**最后更新**: 2026年3月27日  
**状态**: ✅ 训练完成 (10/10 Epochs)  
**下一步**: 推理、评估、部署
