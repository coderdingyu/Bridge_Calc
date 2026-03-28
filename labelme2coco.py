import os
import json
import glob
import numpy as np
import cv2
from pathlib import Path

# ================= 配置区域 =================
DATA_DIR = "/data1/dingyu/datasets/labelme"
SPLITS = ["train", "val", "test"]
CATEGORY_MAPPING = {}
# ============================================

def create_coco_format():
    return {
        "info": {"description": "Bridge Defect Dataset"},
        "images": [],
        "annotations": [],
        "categories": []
    }

def get_image_file(json_file, folder_path):
    """根据 json 文件查找对应的图片文件 - 暴力匹配版（无视大小写，容错强）"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_filename_from_json = data.get("imagePath", "")
    except:
        img_filename_from_json = ""
    
    if img_filename_from_json:
        img_base = os.path.splitext(os.path.basename(img_filename_from_json))[0]
    else:
        img_base = os.path.splitext(os.path.basename(json_file))[0]
    
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = os.path.join(folder_path, img_base + ext)
        if os.path.exists(candidate):
            return os.path.basename(candidate), candidate
        candidate_upper = os.path.join(folder_path, img_base + ext.upper())
        if os.path.exists(candidate_upper):
            return os.path.basename(candidate_upper), candidate_upper
    
    try:
        all_files = os.listdir(folder_path)
        for file in all_files:
            file_base = os.path.splitext(file)[0]
            if file_base.lower() == img_base.lower():
                ext = os.path.splitext(file)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    return file, os.path.join(folder_path, file)
    except:
        pass
    
    return None, None

def convert_to_coco(split_name):
    folder_path = os.path.join(DATA_DIR, split_name)
    if not os.path.exists(folder_path):
        print(f"⚠️  找不到文件夹: {folder_path}，跳过。")
        return

    coco_data = create_coco_format()
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    
    if not json_files:
        print(f"⚠️  {split_name} 文件夹里没有找到 json 文件，跳过。")
        return

    print(f"▶️  开始转换 [{split_name}] 集，共发现 {len(json_files)} 个标注文件...")
    
    ann_id = 1
    skipped_count = 0
    clip_count = 0  # 记录自动修复越界的数量
    
    for img_id, json_file in enumerate(json_files, start=1):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            skipped_count += 1
            continue

        img_filename, img_path = get_image_file(json_file, folder_path)
        if img_filename is None:
            skipped_count += 1
            continue

        h = data.get("imageHeight")
        w = data.get("imageWidth")
        
        if h is None or w is None:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                else:
                    skipped_count += 1
                    continue
            except Exception as e:
                skipped_count += 1
                continue

        coco_data["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": w,
            "height": h
        })

        for shape in data.get("shapes", []):
            try:
                label = shape.get("label", "unknown")
                points = shape.get("points", [])
                shape_type = shape.get("shape_type", "polygon")

                if len(points) < 3 or shape_type not in ["polygon", "line"]:
                    continue

                pts_array = np.array(points, dtype=np.float32)
                
                # ================= 核心修复：强制截断越界坐标 =================
                if np.any(pts_array < 0) or np.any(pts_array[:, 0] > w) or np.any(pts_array[:, 1] > h):
                    clip_count += 1
                    # 把小于0的变成0，把大于宽/高的变成最大边界
                    pts_array[:, 0] = np.clip(pts_array[:, 0], 0, w - 1)
                    pts_array[:, 1] = np.clip(pts_array[:, 1], 0, h - 1)
                # ==============================================================

                if label not in CATEGORY_MAPPING:
                    CATEGORY_MAPPING[label] = len(CATEGORY_MAPPING) + 1

                # 提取修复后的坐标 (必须使用 pts_array 里的新值)
                segmentation = [float(coord) for pt in pts_array for coord in pt]
                
                x_min, y_min = pts_array.min(axis=0)
                x_max, y_max = pts_array.max(axis=0)
                bbox_w = max(1, x_max - x_min)
                bbox_h = max(1, y_max - y_min)
                
                area = cv2.contourArea(pts_array)
                if area < 1:
                    area = 1.0

                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY_MAPPING[label],
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": [float(x_min), float(y_min), float(bbox_w), float(bbox_h)],
                    "iscrowd": 0
                })
                ann_id += 1

            except Exception as e:
                continue

    for label, cat_id in CATEGORY_MAPPING.items():
        if not any(c["id"] == cat_id for c in coco_data["categories"]):
            coco_data["categories"].append({
                "id": cat_id,
                "name": label,
                "supercategory": "defect"
            })

    output_file = os.path.join(DATA_DIR, f"{split_name}_coco.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ [{split_name}] 集转换完成！(自动修复了 {clip_count} 个越界坐标)")
        print(f"   - 图片数量: {len(coco_data['images'])}")
        print(f"   - 标注数量: {ann_id - 1}")
        print(f"   - 跳过文件: {skipped_count}")
        print(f"💾 保存在: {output_file}\n")
    except Exception as e:
        pass

if __name__ == "__main__":
    print("=======================================")
    print("  🚀 Labelme 到 COCO 格式自动转换器 (终极防弹版)")
    print("=======================================\n")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        exit(1)
    
    for split in SPLITS:
        convert_to_coco(split)
        
    print("📊 最终类别字典映射:")
    if CATEGORY_MAPPING:
        for k, v in sorted(CATEGORY_MAPPING.items(), key=lambda x: x[1]):
            print(f"  - ID {v}: {k}")
