#!/usr/bin/env python3
"""
Mask R-CNN 批量推理与可视化脚本 (终极批处理版)
支持单张图片或整个文件夹批量测试，统一输出到项目指定目录
支持量化模块接入（--enable-quant 开启）
"""
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
from PIL import Image
import os
import sys
import glob
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore')

try:
    from bridge_quantify import quantify_predictions, clear_csv_summary
    HAS_QUANTIFY = True
except ImportError:
    HAS_QUANTIFY = False
    print("⚠️ 量化模块 bridge_quantify 未找到，量化功能将不可用")

# ================= 配置区域 =================
NUM_CLASSES = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.5  

DEFAULT_TEST_DIR = "/data1/dingyu/datasets/labelme/test"
OUTPUT_DIR = "/home/dingyu/Bridge_Calc/test_results"

CLASS_NAMES = {0: "Background", 1: "Breakage", 2: "ReinForcement", 3: "Comb", 4: "Crack", 5: "Seepage", 6: "Hole"}
CLASS_COLORS = {0: (0,0,0), 1: (255,0,0), 2: (0,255,0), 3: (0,255,255), 4: (255,0,255), 5: (255,255,0), 6: (255,165,0)}
# ============================================

def load_model(model_path):
    print(f"\n📦 正在加载模型: {model_path}")
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, NUM_CLASSES)
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

def inference(model, image_path):
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)
    image_tensor = torch.from_numpy(
        np.array(image_pil, dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions[0], image_np

def visualize_and_save(predictions, image, output_path):
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy()
    
    detection_summary = {}
    
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < CONFIDENCE_THRESHOLD:
            continue
            
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
        
        mask_binary = (mask > 0.5).astype(np.uint8)
        color = np.array(CLASS_COLORS[label], dtype=np.float32)
        mask_indices = np.where(mask_binary[0])
        if len(mask_indices[0]) > 0:
            result[mask_indices] = result[mask_indices] * 0.4 + color * 0.6
        
        x_min, y_min, x_max, y_max = map(int, box)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), CLASS_COLORS[label], 2)
        
        text = f"{class_name}: {score:.2f}"
        font_scale, thickness = 0.6, 1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        bg_x1, bg_y1 = x_min, max(y_min - 5, 25)
        bg_x2, bg_y2 = min(x_min + text_size[0] + 5, w), bg_y1 + text_size[1] + 5
        cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), CLASS_COLORS[label], -1)
        cv2.putText(result, text, (x_min + 2, bg_y1 + text_size[1] + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return detection_summary

def main():
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print("\n" + "=" * 70)
        print("  🎯 Mask R-CNN 批量推理与可视化 (终极版)")
        print("=" * 70)
        print("\n用法:")
        print("  python inference_final.py <model.pth> [输入路径] [选项]")
        print("\n选项:")
        print("  --enable-quant      启用量化模块")
        print("  --mm-per-pixel      用户输入的像素-物理尺寸转换系数（mm/pixel，不输入时仅输出像素单位）")
        print("  --save-json         保存量化JSON结果")
        print("  --save-csv          保存量化CSV汇总")
        print("  --save-quant-vis    保存量化可视化图")
        print("  --min-mask-area     最小mask面积阈值（默认10）")
        print("\n示例:")
        print("  python inference_final.py models/maskrcnn_bridge_best.pth")
        print("  python inference_final.py models/maskrcnn_bridge_best.pth --enable-quant --save-json")
        print("  python inference_final.py models/maskrcnn_bridge_best.pth --enable-quant --mm-per-pixel 0.5")
        print("=" * 70 + "\n")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Mask R-CNN 批量推理与可视化')
    parser.add_argument('model_path', help='模型路径')
    parser.add_argument('input_path', nargs='?', default=DEFAULT_TEST_DIR, help='输入图片或文件夹路径')
    parser.add_argument('--enable-quant', action='store_true', help='启用量化模块')
    parser.add_argument('--mm-per-pixel', type=float, default=None, help='用户输入的像素-物理尺寸转换系数，单位 mm/pixel；不输入时仅输出像素单位')
    parser.add_argument('--save-json', action='store_true', help='保存量化JSON结果')
    parser.add_argument('--save-csv', action='store_true', help='保存量化CSV汇总')
    parser.add_argument('--save-quant-vis', action='store_true', help='保存量化可视化图')
    parser.add_argument('--min-mask-area', type=int, default=10, help='最小mask面积阈值')
    args = parser.parse_args()
    
    if args.mm_per_pixel is not None and args.mm_per_pixel <= 0:
        print("❌ 参数错误: --mm-per-pixel 必须大于 0")
        sys.exit(1)
    
    model_path = args.model_path
    input_path = args.input_path
    
    if not model_path:
        print("\n❌ 参数错误: 缺少模型路径!"); sys.exit(1)
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}"); sys.exit(1)
    if not os.path.exists(input_path):
        print(f"❌ 输入路径不存在: {input_path}"); sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    quant_enabled = args.enable_quant and HAS_QUANTIFY
    if quant_enabled:
        print("\n🔍 量化模块已启用")
        if args.mm_per_pixel is not None:
            print(f"📏 比例尺: {args.mm_per_pixel} mm/pixel")
        if args.save_csv:
            csv_path = os.path.join(OUTPUT_DIR, 'quant_csv', 'summary.csv')
            clear_csv_summary(csv_path)

    model = load_model(model_path)
    
    image_paths = []
    if os.path.isdir(input_path):
        for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
    else:
        image_paths = [input_path]
        
    if not image_paths:
        print("⚠️ 未找到任何图像文件！"); sys.exit(0)
        
    global_summary = {}
    
    for img_path in tqdm(image_paths, desc="推理进度", ncols=100):
        try:
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(OUTPUT_DIR, f"{name}_detected{ext}")
            
            predictions, image_np = inference(model, img_path)
            summary = visualize_and_save(predictions, image_np, output_path)
            
            if quant_enabled:
                quantify_predictions(
                    predictions=predictions,
                    image=image_np,
                    image_name=basename,
                    mm_per_pixel=args.mm_per_pixel,
                    min_mask_area=args.min_mask_area,
                    crack_min_area=5,
                    save_json=args.save_json,
                    save_csv=args.save_csv,
                    save_vis=args.save_quant_vis,
                    output_dir=OUTPUT_DIR,
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )
            
            for class_name, count in summary.items():
                if class_name not in global_summary:
                    global_summary[class_name] = 0
                global_summary[class_name] += count
        except Exception as e:
            print(f"⚠️ 处理失败 {img_path}: {str(e)}")
            
    print("\n" + "=" * 70)
    print("✨ 批量测试完成！")
    print(f"📂 生成的图像已保存至: {OUTPUT_DIR}")
    
    if quant_enabled and (args.save_json or args.save_csv):
        print(f"📊 量化结果已保存至: {OUTPUT_DIR}/quant_json/ 和 {OUTPUT_DIR}/quant_csv/")
    
    print("\n📈 测试集全局缺陷统计 (共发现 {} 个缺陷):".format(sum(global_summary.values())))
    
    if global_summary:
        for class_name, count in sorted(global_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {class_name}: {count} 个")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
