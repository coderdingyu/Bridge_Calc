#!/usr/bin/env python3
"""
桥梁病害量化计算模块
将 Mask R-CNN 实例分割 mask 转换为工程可用的几何量化结果

支持：
- Crack: 骨架长度 + 局部宽度估计
- 非 Crack: 面积、周长、外接矩形、方向等
- 可选物理单位换算
- JSON/CSV 结构化输出
- 量化可视化
"""
import os
import json
import csv
import numpy as np
import cv2
import warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

warnings.filterwarnings('ignore')

CRACK_CLASS_ID = 4
CLASS_NAMES = {0: "Background", 1: "Breakage", 2: "ReinForcement", 3: "Comb", 4: "Crack", 5: "Seepage", 6: "Hole"}
CLASS_COLORS = {0: (0,0,0), 1: (255,0,0), 2: (0,255,0), 3: (0,255,255), 4: (255,0,255), 5: (255,255,0), 6: (255,165,0)}

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
    import skimage.morphology as skmorph
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def mask_preprocess(mask: np.ndarray, is_crack: bool = False, min_area: int = 10) -> np.ndarray:
    """预处理 mask：二值化、去除极小连通域
    
    注意：对 Crack 类不做形态学操作，因为裂缝线宽可能只有1-2像素，
    开运算会完全破坏裂缝结构
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    if not is_crack:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_binary[labels == i] = 0
    
    return mask_binary


def compute_basic_geometry(mask: np.ndarray) -> Dict[str, Any]:
    """计算基础几何量"""
    if mask.sum() == 0:
        return {
            'area_px': 0, 'perimeter_px': 0, 'bbox_xyxy': [0,0,0,0], 'bbox_xywh': [0,0,0,0],
            'centroid': [0, 0], 'orientation_deg': 0.0, 'major_axis_px': 0.0,
            'minor_axis_px': 0.0, 'compactness': 0.0, 'extent': 0.0
        }
    
    area_px = int(mask.sum())
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        perimeter_px = 0.0
    else:
        contour = max(contours, key=cv2.contourArea)
        perimeter_px = cv2.arcLength(contour, True)
    
    bbox_xywh = list(cv2.boundingRect(mask))
    bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
    
    M = cv2.moments(mask)
    if M['m00'] > 0:
        centroid = [float(M['m10'] / M['m00']), float(M['m01'] / M['m00'])]
    else:
        centroid = [0.0, 0.0]
    
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                _, (MA, ma), angle = cv2.fitEllipse(contour)
                orientation_deg = float(angle) if MA < ma else float(angle + 90)
            else:
                orientation_deg = 0.0
                ma, MA = 0.0, 0.0
        else:
            orientation_deg = 0.0
            MA, ma = 0.0, 0.0
    except Exception:
        orientation_deg, MA, ma = 0.0, 0.0, 0.0
    
    major_axis_px = float(max(MA, ma)) if MA > 0 or ma > 0 else float(max(bbox_xywh[2], bbox_xywh[3]))
    minor_axis_px = float(min(MA, ma)) if MA > 0 or ma > 0 else float(min(bbox_xywh[2], bbox_xywh[3]))
    
    bbox_area = bbox_xywh[2] * bbox_xywh[3]
    extent = float(area_px) / bbox_area if bbox_area > 0 else 0.0
    
    perimeter_sq = perimeter_px ** 2 if perimeter_px > 0 else 1
    compactness = (4 * np.pi * area_px) / perimeter_sq if perimeter_px > 0 else 0.0
    
    return {
        'area_px': area_px, 'perimeter_px': perimeter_px,
        'bbox_xyxy': bbox_xyxy, 'bbox_xywh': bbox_xywh,
        'centroid': centroid, 'orientation_deg': orientation_deg,
        'major_axis_px': major_axis_px, 'minor_axis_px': minor_axis_px,
        'compactness': compactness, 'extent': extent
    }


def compute_crack_metrics(mask: np.ndarray) -> Dict[str, Any]:
    """计算 Crack 类裂缝的精细量化指标：骨架长度 + 局部宽度"""
    metrics = {
        'length_px': 0.0, 'mean_width_px': 0.0, 'max_width_px': 0.0,
        'p95_width_px': 0.0, 'skeleton_pixel_count': 0,
        'branch_count': 0, 'measurement_method': 'skeleton_distance_transform'
    }
    
    if mask.sum() == 0:
        return {**metrics, 'warnings': ['empty mask']}
    
    if HAS_SKIMAGE:
        try:
            skeleton = skeletonize(mask > 0, method='lee')
        except Exception:
            try:
                skeleton = skeletonize(mask > 0)
            except Exception:
                skeleton = _skeletonize_fallback(mask)
    else:
        skeleton = _skeletonize_fallback(mask)
    
    if skeleton is None or skeleton.sum() == 0:
        return {**metrics, 'warnings': ['skeleton extraction failed']}
    
    skeleton_binary = (skeleton > 0).astype(np.uint8)
    skeleton_pixel_count = int(skeleton_binary.sum())
    metrics['skeleton_pixel_count'] = skeleton_pixel_count
    
    length_px = _compute_skeleton_length(skeleton_binary)
    metrics['length_px'] = length_px
    
    width_map = _compute_distance_transform(mask)
    if width_map is not None and width_map.size > 0:
        skeleton_mask = skeleton_binary > 0
        widths_at_skeleton = width_map[skeleton_mask]
        full_widths = 2.0 * widths_at_skeleton[widths_at_skeleton > 0]

        if len(full_widths) > 0:
            metrics['mean_width_px'] = float(np.mean(full_widths))
            metrics['max_width_px'] = float(np.max(full_widths))
            metrics['p95_width_px'] = float(np.percentile(full_widths, 95))
    
    branch_count = _count_skeleton_branches(skeleton_binary)
    metrics['branch_count'] = branch_count
    
    return metrics


def _skeletonize_fallback(mask: np.ndarray) -> Optional[np.ndarray]:
    """无 skimage 时的骨架提取 fallback"""
    try:
        binary = (mask > 0).astype(np.uint8)
        size = max(binary.shape)
        if size < 3:
            return None
        
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(binary)
        temp = binary.copy()
        
        for _ in range(size * 2):
            opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
            temp_diff = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, temp_diff)
            temp = cv2.erode(temp, element)
            if cv2.countNonZero(temp) == 0:
                break
        
        return skeleton
    except Exception:
        return None


def _compute_skeleton_length(skeleton: np.ndarray) -> float:
    """计算骨架长度（8邻接图，水平垂直=1，对角=sqrt(2)，每条边只计一次）"""
    try:
        skel = (skeleton > 0).astype(np.uint8)
        h, w = skel.shape
        
        length = 0.0
        for y in range(h):
            for x in range(w):
                if skel[y, x] == 0:
                    continue
                
                if y < h - 1 and skel[y + 1, x] > 0:
                    length += 1.0
                if x < w - 1 and skel[y, x + 1] > 0:
                    length += 1.0
                if y < h - 1 and x < w - 1 and skel[y + 1, x + 1] > 0:
                    length += np.sqrt(2)
                if y < h - 1 and x > 0 and skel[y + 1, x - 1] > 0:
                    length += np.sqrt(2)
        
        return length
    except Exception:
        return float(skeleton.sum())


def _compute_distance_transform(mask: np.ndarray) -> Optional[np.ndarray]:
    """计算距离变换获取宽度估计"""
    try:
        binary = (mask > 0).astype(np.uint8)
        if HAS_SCIPY:
            dist = ndimage.distance_transform_edt(binary)
        else:
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        return dist
    except Exception:
        return None


def _count_skeleton_branches(skeleton: np.ndarray) -> int:
    """计算骨架分支数（端点数量近似）"""
    try:
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        filtered = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
        branch_points = np.sum(filtered > 12)
        
        h, w = skeleton.shape
        endpoints = 0
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] > 0:
                    neighbors = sum([
                        skeleton[y - 1, x] > 0, skeleton[y + 1, x] > 0,
                        skeleton[y, x - 1] > 0, skeleton[y, x + 1] > 0,
                        skeleton[y - 1, x - 1] > 0, skeleton[y - 1, x + 1] > 0,
                        skeleton[y + 1, x - 1] > 0, skeleton[y + 1, x + 1] > 0
                    ])
                    if neighbors == 1:
                        endpoints += 1
        
        branches = endpoints // 2 + branch_points
        return branches
    except Exception:
        return 0


def compute_region_metrics(mask: np.ndarray) -> Dict[str, Any]:
    """计算非 Crack 类面状病害的量化指标"""
    basic = compute_basic_geometry(mask)
    
    if basic['area_px'] == 0:
        return {**basic, 'min_area_rect': [0, 0, 0, 0, 0.0]}

    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)
            min_area_rect = [float(v) for v in list(rect[0]) + list(rect[1])] + [float(rect[2])]
        else:
            min_area_rect = [0, 0, 0, 0, 0.0]
    except Exception:
        min_area_rect = [0, 0, 0, 0, 0.0]
    
    return {
        **basic,
        'min_area_rect': min_area_rect
    }


def convert_physical_units(metrics: Dict[str, Any], mm_per_pixel: Optional[float] = None) -> Dict[str, Any]:
    """物理单位换算（可选）"""
    if mm_per_pixel is None or mm_per_pixel <= 0:
        return metrics
    
    result = {**metrics}
    mmpp = float(mm_per_pixel)
    
    for key in ['area_px']:
        if key in result and result[key] > 0:
            result[key.replace('_px', '_mm2')] = round(result[key] * mmpp * mmpp, 4)
    
    for key in ['perimeter_px', 'major_axis_px', 'minor_axis_px', 'length_px', 'mean_width_px', 'max_width_px', 'p95_width_px']:
        if key in result and result[key] > 0:
            result[key.replace('_px', '_mm')] = round(result[key] * mmpp, 4)
    
    return result


def estimate_measurement_confidence(
    metrics: Dict[str, Any],
    mask: np.ndarray,
    score: float,
    class_name: str,
    img_h: int,
    img_w: int
) -> Tuple[float, List[str]]:
    """估计测量置信度"""
    warnings_list = []
    confidence = float(score)
    
    area = metrics.get('area_px', 0)
    if area < 100:
        confidence *= 0.7
        warnings_list.append('mask area too small')
    elif area < 500:
        confidence *= 0.9
    
    bbox = metrics.get('bbox_xyxy', [0,0,0,0])
    margin = 5
    if bbox[0] < margin or bbox[1] < margin or bbox[2] > img_w - margin or bbox[3] > img_h - margin:
        confidence *= 0.8
        warnings_list.append('mask touches image boundary')
    
    if class_name == 'Crack':
        length = metrics.get('length_px', 0)
        if length < 50:
            confidence *= 0.7
            warnings_list.append('crack length too small for reliable measurement')
        
        if metrics.get('measurement_method') == 'skeleton_distance_transform':
            if metrics.get('mean_width_px', 0) < 1:
                confidence *= 0.6
                warnings_list.append('width measurement may be unreliable')
    
    compactness = metrics.get('compactness', 1)
    if class_name != 'Crack' and compactness < 0.1:
        confidence *= 0.8
        warnings_list.append('highly irregular shape')
    
    confidence = max(0.1, min(1.0, confidence))
    return round(confidence, 4), warnings_list


def quantify_instance(
    mask: np.ndarray,
    class_id: int,
    class_name: str,
    score: float,
    instance_id: int,
    image_name: str,
    bbox: List[float],
    mm_per_pixel: Optional[float] = None,
    min_mask_area: int = 10,
    crack_min_area: int = 5,
    img_h: int = 0,
    img_w: int = 0
) -> Dict[str, Any]:
    """量化单个实例"""
    is_crack = (class_id == CRACK_CLASS_ID)

    min_area = crack_min_area if is_crack else min_mask_area
    mask_clean = mask_preprocess(mask, is_crack=is_crack, min_area=min_area)
    
    if mask_clean.sum() == 0:
        return {
            'image_name': image_name, 'instance_id': instance_id, 'class_id': class_id,
            'class_name': class_name, 'score': round(float(score), 4), 'bbox': bbox,
            'error': 'mask too small after preprocessing', 'measurement_confidence': 0.0
        }
    
    if is_crack:
        geometry = compute_basic_geometry(mask_clean)
        crack_metrics = compute_crack_metrics(mask_clean)
        metrics = {**geometry, **crack_metrics}
    else:
        metrics = compute_region_metrics(mask_clean)
    
    metrics = convert_physical_units(metrics, mm_per_pixel)
    
    confidence, conf_warnings = estimate_measurement_confidence(
        metrics, mask_clean, score, class_name, img_h, img_w
    )
    
    result = {
        'image_name': image_name, 'instance_id': instance_id,
        'class_id': class_id, 'class_name': class_name,
        'score': round(float(score), 4), 'bbox': [float(x) for x in bbox],
        'measurement_confidence': confidence,
        'warnings': conf_warnings
    }
    result.update(metrics)
    
    return result


def save_json_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """保存 JSON 结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def clear_csv_summary(csv_path: str) -> None:
    """清空 CSV 文件（整批任务开始前调用）"""
    if os.path.exists(csv_path):
        os.remove(csv_path)


def append_csv_summary(results: List[Dict[str, Any]], csv_path: str, append: bool = True) -> None:
    """保存 CSV 汇总

    Args:
        results: 量化结果列表
        csv_path: CSV 文件路径
        append: True=追加模式, False=覆盖模式
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        'image_name', 'instance_id', 'class_id', 'class_name', 'score',
        'bbox_xyxy', 'area_px', 'perimeter_px', 'length_px', 'mean_width_px',
        'max_width_px', 'p95_width_px', 'major_axis_px', 'minor_axis_px',
        'orientation_deg', 'compactness', 'extent', 'measurement_confidence',
        'warnings', 'length_mm', 'mean_width_mm', 'area_mm2'
    ]

    mode = 'a' if append else 'w'
    file_exists = os.path.exists(csv_path) and append

    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            if 'warnings' in r and isinstance(r['warnings'], list):
                row['warnings'] = '; '.join(r['warnings'])
            if 'bbox_xyxy' in r:
                row['bbox_xyxy'] = '; '.join([f"{x:.1f}" for x in r['bbox_xyxy']])
            writer.writerow(row)


def visualize_quantitative(
    image: np.ndarray,
    results: List[Dict[str, Any]],
    output_path: str
) -> None:
    """生成量化可视化图"""
    vis = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    
    for r in results:
        if 'error' in r:
            continue
        
        class_name = r['class_name']
        score = r['score']
        conf = r.get('measurement_confidence', 1.0)
        class_id = r['class_id']
        
        mask_area = r.get('area_px', 0)
        text_lines = [f"{class_name}", f"S:{score:.2f} C:{conf:.2f}", f"A:{mask_area}px"]
        
        if class_id == CRACK_CLASS_ID:
            length = r.get('length_px', 0)
            mean_w = r.get('mean_width_px', 0)
            if length > 0:
                text_lines.append(f"L:{length:.0f} W:{mean_w:.1f}")
            if 'length_mm' in r and 'mean_width_mm' in r:
                text_lines.append(f"L:{r['length_mm']:.1f}mm W:{r['mean_width_mm']:.2f}mm")
        else:
            if 'area_mm2' in r:
                text_lines.append(f"S:{r['area_mm2']:.2f}mm2")
        
        bbox = r['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        color = np.array(CLASS_COLORS.get(class_id, (255, 255, 255)), dtype=np.float32)
        cv2.rectangle(vis, (x1, y1), (x2, y2), tuple(color.tolist()), 2)
        
        font_scale = 0.5
        thickness = 1
        y_offset = y1 - 5
        for line in reversed(text_lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis, (x1, max(0, y_offset - th - 3)), (x1 + tw + 5, y_offset + 3), tuple((color * 0.8).tolist()), -1)
            cv2.putText(vis, line, (x1 + 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            y_offset -= (th + 5)
    
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def quantify_predictions(
    predictions: Dict[str, np.ndarray],
    image: np.ndarray,
    image_name: str,
    mm_per_pixel: Optional[float] = None,
    min_mask_area: int = 10,
    crack_min_area: int = 5,
    save_json: bool = False,
    save_csv: bool = False,
    save_vis: bool = False,
    output_dir: str = "/home/dingyu/Bridge_Calc/test_results",
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """量化一组预测结果"""
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy()

    h, w = image.shape[:2]
    results = []

    for idx, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        if score < confidence_threshold:
            continue
        
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        class_id = int(label)
        
        try:
            result = quantify_instance(
                mask=mask[0],
                class_id=class_id,
                class_name=class_name,
                score=score,
                instance_id=idx,
                image_name=image_name,
                bbox=box.tolist(),
                mm_per_pixel=mm_per_pixel,
                min_mask_area=min_mask_area,
                crack_min_area=crack_min_area,
                img_h=h, img_w=w
            )
            results.append(result)
        except Exception as e:
            results.append({
                'image_name': image_name, 'instance_id': idx,
                'class_id': class_id, 'class_name': class_name,
                'score': round(float(score), 4), 'bbox': box.tolist(),
                'error': str(e), 'measurement_confidence': 0.0
            })
    
    if save_json:
        json_dir = os.path.join(output_dir, 'quant_json')
        json_path = os.path.join(json_dir, f"{os.path.splitext(image_name)[0]}_quant.json")
        save_json_results(results, json_path)
    
    if save_csv:
        csv_path = os.path.join(output_dir, 'quant_csv', 'summary.csv')
        append_csv_summary(results, csv_path)
    
    if save_vis:
        vis_dir = os.path.join(output_dir, 'quant_vis')
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{os.path.splitext(image_name)[0]}_quant_vis.jpg")
        visualize_quantitative(image, results, vis_path)
    
    return results
