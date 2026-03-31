#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 Labelme 的人工标注（GT）可视化成和彩色预测图类似的效果：
- 彩色 mask 填充
- bbox 矩形框
- 类别文字

用法：
python visualize_gt_labelme.py \
    --image /data1/dingyu/datasets/labelme/test/xxx.jpg \
    --json  /data1/dingyu/datasets/labelme/test/xxx.json \
    --out   /data1/dingyu/datasets/labelme/test_gt_vis/xxx_gt.jpg
"""

import os
import json
import argparse
from typing import Dict, Tuple, List

import cv2
import numpy as np


CLASS_COLORS = {
    "Breakage": (255, 0, 0),
    "ReinForcement": (0, 255, 0),
    "Comb": (0, 255, 255),
    "Crack": (255, 0, 255),
    "Seepage": (255, 255, 0),
    "Hole": (255, 165, 0),
}
DEFAULT_COLOR = (255, 255, 255)


def load_labelme_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def polygon_to_mask(points: List[List[float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def draw_gt(
    image_rgb: np.ndarray,
    labelme_data: dict,
    alpha: float = 0.45,
    draw_bbox: bool = True,
    draw_label: bool = True
) -> np.ndarray:
    result = image_rgb.copy().astype(np.float32)
    h, w = image_rgb.shape[:2]

    shapes = labelme_data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label", "unknown")
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        if shape_type != "polygon" or len(points) < 3:
            continue

        color = np.array(CLASS_COLORS.get(label, DEFAULT_COLOR), dtype=np.float32)

        mask = polygon_to_mask(points, h, w)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue

        result[ys, xs] = result[ys, xs] * (1 - alpha) + color * alpha

        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))

        if draw_bbox:
            cv2.rectangle(
                result,
                (x_min, y_min),
                (x_max, y_max),
                tuple(color.tolist()),
                2
            )

        if draw_label:
            text = f"GT: {label}"
            font_scale = 0.6
            thickness = 1
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            bg_x1 = x_min
            bg_y1 = max(y_min - 5, 25)
            bg_x2 = min(x_min + tw + 6, w - 1)
            bg_y2 = min(bg_y1 + th + 6, h - 1)

            cv2.rectangle(
                result,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                tuple(color.tolist()),
                -1
            )
            cv2.putText(
                result,
                text,
                (bg_x1 + 2, bg_y1 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize Labelme GT")
    parser.add_argument("--image", required=True, help="原始图片路径")
    parser.add_argument("--json", required=True, help="对应的 Labelme JSON 路径")
    parser.add_argument("--out", required=True, help="输出可视化图片路径")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 透明度，默认 0.45")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"找不到图片: {args.image}")
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"找不到 JSON: {args.json}")

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"无法读取图片: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    labelme_data = load_labelme_json(args.json)
    vis_rgb = draw_gt(image_rgb, labelme_data, alpha=args.alpha)

    ensure_dir(args.out)
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(args.out, vis_bgr)
    if not ok:
        raise RuntimeError(f"保存失败: {args.out}")

    print(f"✅ GT 可视化已保存到: {args.out}")


if __name__ == "__main__":
    main()
