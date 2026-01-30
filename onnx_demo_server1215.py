#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 可视化演示服务 (增强版)

功能:
- 批量检测整个目录下的图片
- 计算检测正确率（基于IoU匹配）
- 自动筛选高正确率图片并保存
- 支持原图、预测图、真实标签图对比展示

使用方法:
    直接运行: python onnx_demo_server.py
"""

import os
import sys
import base64
import time
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ============================================================
# ★★★ 配置区域 - 在此处修改配置 ★★★
# ============================================================

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8082

MODEL_DIR = './'
IMAGE_DIR = './test_images'

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# ============================================================

ALGO_CONFIG = {
    1: {"name": "松线虫害识别", "classes": {0: "死亡", 1: "重度患病", 2: "轻度患病"}},
    2: {"name": "河道淤积识别", "classes": {0: "水污染", 1: "漂浮碎片", 2: "废弃船只", 3: "渔业和水产养殖", 4: "垃圾"}},
    3: {"name": "漂浮物识别",
        "classes": {0: "瓶子", 1: "草", 2: "树枝", 3: "牛奶盒", 4: "塑料袋", 5: "塑料垃圾袋", 6: "球", 7: "叶子"}},
    4: {"name": "游泳涉水识别",
        "classes": {0: "忽略", 1: "游泳者", 2: "船", 3: "水上摩托艇", 4: "救生设备", 5: "浮标"}},
    5: {"name": "车牌识别", "classes": {0: "车牌"}},
    6: {"name": "交通拥堵识别", "classes": {0: "车辆"}},
    7: {"name": "路面破损识别",
        "classes": {0: "龟裂", 1: "纵向裂缝", 2: "纵向修补块", 3: "检查井井盖", 4: "坑洞", 5: "横向裂缝",
                    6: "横向修补块"}},
    8: {"name": "路面污染",
        "classes": {0: "裂缝", 1: "积水", 2: "路面松散", 3: "泥泞道路", 4: "路边垃圾", 5: "坑洞"}},
    9: {"name": "人群聚集识别", "classes": {0: "车", 1: "人"}},
    10: {"name": "非法垂钓识别", "classes": {0: "水边钓鱼", 1: "游泳溺水", 2: "钓鱼伞", 3: "船"}},
    11: {"name": "施工识别", "classes": {0: "起重机", 1: "挖掘机", 2: "拖拉机", 3: "卡车"}},
    12: {"name": "秸秆焚烧", "classes": {0: "秸秆堆"}},
    13: {"name": "变化检测", "classes": {0: "无变化", 1: "变化区域"}},
    14: {"name": "占道经营识别", "classes": {0: "占道经营"}},
    15: {"name": "垃圾堆放识别",
         "classes": {0: "长椅", 1: "商业垃圾", 2: "非法倾倒点", 3: "绿地", 4: "孔洞", 5: "泽西护栏", 6: "地块",
                     7: "原材料", 8: "生活垃圾"}},
    16: {"name": "裸土未覆盖识别", "classes": {0: "垃圾", 1: "裸土"}},
    17: {"name": "建控区违建识别", "classes": {0: "蓝色天篷", 1: "其他违建", 2: "改装绿色小屋"}},
    18: {"name": "烟火识别", "classes": {0: "烟雾", 1: "火"}},
    19: {"name": "光伏板缺陷检测", "classes": {0: "有缺陷的光伏电池"}},
    20: {"name": "园区夜间入侵检测", "classes": {0: "人", 1: "车", 2: "自行车"}},
    21: {"name": "园区外立面病害识别",
         "classes": {0: "墙体腐蚀", 1: "墙体开裂", 2: "墙体劣化", 3: "墙模", 4: "墙面污渍"}},
    22: {"name": "罂粟识别", "classes": {0: "罂粟"}},
    23: {"name": "作物倒伏检测", "classes": {0: "作物倒伏"}},
    24: {"name": "林业侵占",
         "classes": {0: "反铲装载机", 1: "压路机", 2: "混凝土搅拌车", 3: "推土机", 4: "倾卸卡车", 5: "挖掘机",
                     6: "平地机", 7: "安全头盔", 8: "移动式起重机", 9: "人", 10: "塔式起重机", 11: "背心",
                     12: "轮式装载机"}},
    999: {"name": "人脸检测", "classes": {0: "人脸"}},
}

MODEL_TYPES = {
    7: "yolov11_720", 8: "yolov11_720", 10: "yolov11_720", 12: "yolov11_720",
    14: "yolov11_720", 15: "yolov11_720", 16: "yolov11_720", 18: "yolov11_720",
    6: "yolov11",
    13: "change_detection",
}


# ============================================================
# 正确率计算工具
# ============================================================
def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Dict]:
    """解析YOLO格式的标签文件"""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx = float(parts[1]) * img_width
                    cy = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height
                    
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2]
                    })
    except Exception as e:
        print(f"[警告] 解析标签文件失败: {label_path}, {e}")
    
    return labels


def calculate_accuracy(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    计算检测正确率
    正确率 = 正确匹配数 / max(预测数, 真实数)
    """
    if len(ground_truths) == 0 and len(predictions) == 0:
        return 1.0
    
    if len(ground_truths) == 0 or len(predictions) == 0:
        return 0.0
    
    matched_gt = set()
    matched_pred = set()
    
    # 贪婪匹配：按IoU从高到低匹配
    matches = []
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            if pred.get('class_id', -1) == gt.get('class_id', -2):
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold:
                    matches.append((iou, i, j))
    
    # 按IoU排序
    matches.sort(reverse=True)
    
    # 贪婪选择
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    correct_count = len(matched_gt)
    total = max(len(predictions), len(ground_truths))
    
    return correct_count / total if total > 0 else 0.0


def detections_to_yolo_format(detections: List[Dict], img_width: int, img_height: int) -> str:
    """将检测结果转换为YOLO格式的标签字符串"""
    lines = []
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        
        class_id = det['class_id']
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    return '\n'.join(lines)


# ============================================================
# ONNX 变化检测器
# ============================================================
class ONNXChangeDetector:
    """ONNX 变化检测器 (SNUNet)"""

    def __init__(self, model_path: str, algo_id: int = 13):
        self.model_path = model_path
        self.algo_id = algo_id
        self.input_size = 256

        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[ONNX] 加载变化检测模型: {model_path}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        img_norm = img_chw.astype(np.float32) / 255.0
        return np.expand_dims(img_norm, axis=0)

    def detect(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float, float]:
        orig_h, orig_w = image1.shape[:2]
        input1 = self.preprocess(image1)
        input2 = self.preprocess(image2)

        t0 = time.time()

        if len(self.input_names) == 2:
            inputs = {self.input_names[0]: input1, self.input_names[1]: input2}
        else:
            combined = np.concatenate([input1, input2], axis=1)
            inputs = {self.input_names[0]: combined}

        outputs = self.session.run(None, inputs)
        infer_time = time.time() - t0

        output = outputs[0]

        if output.ndim == 4 and output.shape[1] == 2:
            pred = np.argmax(output, axis=1)[0]
            mask_binary = (pred == 1).astype(np.uint8) * 255
        elif output.ndim == 4 and output.shape[1] == 1:
            mask_binary = (output[0, 0] > 0).astype(np.uint8) * 255
        elif output.ndim == 3:
            if output.shape[0] == 2:
                pred = np.argmax(output, axis=0)
                mask_binary = (pred == 1).astype(np.uint8) * 255
            else:
                mask_binary = (output[0] > 0).astype(np.uint8) * 255
        else:
            mask_binary = (output > 0).astype(np.uint8) * 255
            if mask_binary.ndim > 2:
                mask_binary = mask_binary.squeeze()

        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        change_ratio = np.sum(mask_resized > 0) / (orig_h * orig_w)

        mask_color = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        mask_color[mask_resized > 0] = [0, 0, 255]

        return mask_color, change_ratio, infer_time


# ============================================================
# ONNX YOLO 检测器
# ============================================================
class ONNXYOLODetector:
    """ONNX YOLO 检测器 (兼容 YOLOv5/v11)"""

    def __init__(self, model_path: str, algo_id: int):
        self.model_path = model_path
        self.algo_id = algo_id
        self.model_type = MODEL_TYPES.get(algo_id, "yolov5")

        if self.model_type == "yolov11_720":
            self.input_size = 736
        elif self.model_type == "change_detection":
            self.input_size = 256
        else:
            self.input_size = 640

        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        output_shape = self.session.get_outputs()[0].shape
        self._detect_model_format(output_shape)

        print(f"[ONNX] 加载模型: algo={algo_id}, path={model_path}")
        print(f"       类型={self.model_type}, 输入={self.input_size}, 格式={self._format}")

    def _detect_model_format(self, output_shape):
        if len(output_shape) == 3:
            dim1 = output_shape[1]
            dim2 = output_shape[2]

            if isinstance(dim1, int) and isinstance(dim2, int):
                if dim1 > dim2:
                    self._format = "yolov5"
                    self._has_objectness = True
                    self._transpose = False
                else:
                    self._format = "yolov11"
                    self._has_objectness = False
                    self._transpose = True
            else:
                if self.model_type in ("yolov11", "yolov11_720"):
                    self._format = "yolov11"
                    self._has_objectness = False
                    self._transpose = True
                else:
                    self._format = "yolov5"
                    self._has_objectness = True
                    self._transpose = False
        else:
            self._format = "yolov5"
            self._has_objectness = True
            self._transpose = False

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, tuple]:
        orig_h, orig_w = image.shape[:2]
        input_size = self.input_size

        r = min(input_size / orig_h, input_size / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)

        img_resized = cv2.resize(image, (new_w, new_h))

        dw, dh = (input_size - new_w) / 2, (input_size - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        img_norm = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        return img_batch, (orig_h, orig_w, r, top, left)

    def detect(self, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[Dict]:
        input_data, img_info = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0]
        return self._postprocess(output, img_info, conf_thres, iou_thres)

    def _postprocess(self, output: np.ndarray, img_info: tuple,
                     conf_thres: float, iou_thres: float) -> List[Dict]:
        orig_h, orig_w, ratio, pad_top, pad_left = img_info

        if output.ndim == 3:
            output = output[0]

        if self._transpose:
            predictions = output.T
        else:
            predictions = output

        boxes = predictions[:, :4].copy()

        if self._has_objectness:
            obj_conf = predictions[:, 4]
            class_scores = predictions[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            class_conf = np.max(class_scores, axis=1)
            scores = obj_conf * class_conf
        else:
            class_scores = predictions[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = np.max(class_scores, axis=1)

        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        boxes_xyxy = np.copy(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= ratio

        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes_xyxy) == 0:
            return []

        keep = self._nms(boxes_xyxy, scores, iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        detections = []
        algo_classes = ALGO_CONFIG.get(self.algo_id, {}).get("classes", {})

        for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
            x1, y1, x2, y2 = box
            class_name = algo_classes.get(int(cls_id), f"class_{cls_id}")
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(score),
                "class_id": int(cls_id),
                "class_name": class_name
            })

        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep


# ============================================================
# 模型管理器
# ============================================================
class ModelManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models: Dict[int, Any] = {}
        self._scan_models()

    def _scan_models(self):
        if not os.path.exists(self.model_dir):
            print(f"[警告] 模型目录不存在: {self.model_dir}")
            return

        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.onnx'):
                continue

            filepath = os.path.join(self.model_dir, filename)
            base_name = filename.replace('.onnx', '')
            base_name = base_name.replace('_720', '').replace('_bs1', '')

            try:
                algo_id = int(base_name)
            except ValueError:
                continue

            if '_720' in filename and algo_id in self.models:
                pass
            elif algo_id in self.models:
                continue

            try:
                model_type = MODEL_TYPES.get(algo_id, "yolov5")
                if model_type == "change_detection":
                    self.models[algo_id] = ONNXChangeDetector(filepath, algo_id)
                else:
                    self.models[algo_id] = ONNXYOLODetector(filepath, algo_id)
            except Exception as e:
                print(f"[错误] 加载模型失败: {filename}, {e}")

        print(f"[模型管理器] 加载了 {len(self.models)} 个模型")

    def get_detector(self, algo_id: int):
        return self.models.get(algo_id)

    def is_change_detection(self, algo_id: int) -> bool:
        return MODEL_TYPES.get(algo_id) == "change_detection"

    def get_available_algos(self) -> List[Dict]:
        algos = []
        for algo_id in sorted(self.models.keys()):
            config = ALGO_CONFIG.get(algo_id, {})
            algos.append({
                "id": algo_id,
                "name": config.get("name", f"算法{algo_id}"),
                "classes": config.get("classes", {}),
                "type": "change_detection" if self.is_change_detection(algo_id) else "detection"
            })
        return algos


# ============================================================
# 图片文件夹管理器
# ============================================================
class ImageFolderManager:
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, image_dir: str):
        self.image_dir = image_dir

    def get_folders(self) -> List[Dict]:
        if not os.path.exists(self.image_dir):
            return []

        folders = []

        root_image_count = sum(1 for f in os.listdir(self.image_dir)
                               if os.path.isfile(os.path.join(self.image_dir, f)) and
                               Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS)
        if root_image_count > 0:
            folders.append({
                "name": "📷 当前目录",
                "path": "__ROOT__",
                "image_count": root_image_count
            })

        for name in sorted(os.listdir(self.image_dir)):
            folder_path = os.path.join(self.image_dir, name)
            if os.path.isdir(folder_path):
                image_count = sum(1 for f in os.listdir(folder_path)
                                  if Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS)
                if image_count > 0:
                    folders.append({
                        "name": name,
                        "path": name,
                        "image_count": image_count
                    })

        return folders

    def get_images(self, folder_name: str) -> List[Dict]:
        if folder_name == "__ROOT__":
            folder_path = self.image_dir
        else:
            folder_path = os.path.join(self.image_dir, folder_name)

        if not os.path.exists(folder_path):
            return []

        images = []
        for filename in sorted(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath) and Path(filename).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                size = os.path.getsize(filepath)
                if folder_name == "__ROOT__":
                    rel_path = filename
                else:
                    rel_path = f"{folder_name}/{filename}"
                images.append({
                    "name": filename,
                    "path": rel_path,
                    "size": size,
                    "size_str": self._format_size(size)
                })

        return images

    def get_all_images(self) -> List[Dict]:
        """获取目录下所有图片（不递归）"""
        if not os.path.exists(self.image_dir):
            return []

        images = []
        for filename in sorted(os.listdir(self.image_dir)):
            filepath = os.path.join(self.image_dir, filename)
            if os.path.isfile(filepath) and Path(filename).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                size = os.path.getsize(filepath)
                images.append({
                    "name": filename,
                    "path": filename,
                    "full_path": filepath,
                    "size": size,
                    "size_str": self._format_size(size)
                })

        return images

    def get_image_path(self, relative_path: str) -> Optional[str]:
        full_path = os.path.join(self.image_dir, relative_path)
        if os.path.exists(full_path):
            return full_path
        return None

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


# ============================================================
# 绘图工具
# ============================================================
def draw_detections(image: np.ndarray, detections: List[Dict], algo_id: int = None) -> np.ndarray:
    """在图片上绘制检测结果（支持中文）"""
    from PIL import Image, ImageDraw, ImageFont

    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_image)

    font = None
    font_size = 20
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    ]

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = det.get("class_id", 0)
        class_name = det.get("class_name", f"class_{class_id}")
        confidence = det.get("confidence", 1.0)

        color = colors[class_id % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{class_name} {confidence:.2f}"

        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except:
            text_w, text_h = len(label) * 10, 20

        label_y = max(y1 - text_h - 8, 0)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 6], fill=color)
        draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255), font=font)

    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def draw_ground_truth(image: np.ndarray, labels: List[Dict], algo_id: int = None) -> np.ndarray:
    """根据真实标签绘制框"""
    from PIL import Image, ImageDraw, ImageFont

    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_image)

    font = None
    font_size = 20
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # 使用不同的颜色方案（蓝色系）来区分真实标签
    colors = [
        (0, 128, 255), (0, 200, 255), (50, 150, 255), (100, 100, 255),
        (150, 50, 255), (200, 0, 200), (255, 100, 150), (100, 255, 200),
    ]

    algo_classes = ALGO_CONFIG.get(algo_id, {}).get("classes", {}) if algo_id else {}

    for label in labels:
        bbox = label["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = label.get("class_id", 0)
        class_name = algo_classes.get(class_id, f"class_{class_id}")

        color = colors[class_id % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = f"GT: {class_name}"

        try:
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except:
            text_w, text_h = len(label_text) * 10, 20

        label_y = max(y1 - text_h - 8, 0)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 6], fill=color)
        draw.text((x1 + 5, label_y + 2), label_text, fill=(255, 255, 255), font=font)

    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


# ============================================================
# Flask 应用
# ============================================================
app = Flask(__name__, static_folder=None)
CORS(app)

model_manager: ModelManager = None
image_manager: ImageFolderManager = None

# 存储批量检测结果
batch_results = {}


@app.route('/')
def index():
    return send_from_directory('.', 'onnx_demo.html')


@app.route('/api/config', methods=['POST', 'GET'])
def api_config():
    global model_manager, image_manager

    if request.method == 'GET':
        return jsonify({
            "code": 200,
            "data": {
                "model_dir": model_manager.model_dir if model_manager else MODEL_DIR,
                "image_dir": image_manager.image_dir if image_manager else IMAGE_DIR
            }
        })

    try:
        data = request.get_json()
        new_model_dir = data.get('model_dir')
        new_image_dir = data.get('image_dir')

        if new_model_dir:
            if os.path.exists(new_model_dir):
                model_manager = ModelManager(new_model_dir)
                print(f"[配置] 模型目录已更新: {new_model_dir}")
            else:
                return jsonify({"code": 400, "message": f"模型目录不存在: {new_model_dir}"})

        if new_image_dir:
            if os.path.exists(new_image_dir):
                image_manager = ImageFolderManager(new_image_dir)
                print(f"[配置] 图片目录已更新: {new_image_dir}")
            else:
                return jsonify({"code": 400, "message": f"图片目录不存在: {new_image_dir}"})

        return jsonify({
            "code": 200,
            "message": "配置已更新",
            "data": {
                "model_dir": model_manager.model_dir if model_manager else new_model_dir,
                "image_dir": image_manager.image_dir if image_manager else new_image_dir
            }
        })
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)})


@app.route('/api/algorithms')
def get_algorithms():
    return jsonify({
        "code": 200,
        "data": model_manager.get_available_algos()
    })


@app.route('/api/folders')
def get_folders():
    return jsonify({
        "code": 200,
        "data": image_manager.get_folders()
    })


@app.route('/api/images/<path:folder_name>')
def get_images(folder_name: str):
    return jsonify({
        "code": 200,
        "data": image_manager.get_images(folder_name)
    })


@app.route('/api/image/<path:image_path>')
def get_image(image_path: str):
    full_path = image_manager.get_image_path(image_path)
    if full_path:
        return send_file(full_path)
    return jsonify({"code": 404, "message": "图片不存在"}), 404


@app.route('/api/thumb/<path:image_path>')
def get_thumbnail(image_path: str):
    full_path = image_manager.get_image_path(image_path)
    if not full_path:
        return jsonify({"code": 404, "message": "图片不存在"}), 404

    try:
        image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"code": 400, "message": "无法读取图片"}), 400

        h, w = image.shape[:2]
        max_size = 100
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

        thumb = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])

        return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect():
    """执行单张图片目标检测"""
    data = request.json

    algo_id = data.get('algorithm_id')
    image_path = data.get('image_path')
    conf_threshold = data.get('conf_threshold', DEFAULT_CONF)
    iou_threshold = data.get('iou_threshold', DEFAULT_IOU)

    if not algo_id:
        return jsonify({"code": 400, "message": "缺少 algorithm_id"}), 400

    if not image_path:
        return jsonify({"code": 400, "message": "缺少 image_path"}), 400

    detector = model_manager.get_detector(algo_id)
    if not detector:
        return jsonify({"code": 404, "message": f"算法 {algo_id} 不存在"}), 404

    full_path = image_manager.get_image_path(image_path)
    if not full_path:
        return jsonify({"code": 404, "message": "图片不存在"}), 404

    try:
        image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400

    if image is None:
        return jsonify({"code": 400, "message": "无法读取图片"}), 400

    img_h, img_w = image.shape[:2]

    t0 = time.time()
    try:
        detections = detector.detect(image, conf_threshold, iou_threshold)
    except Exception as e:
        return jsonify({"code": 500, "message": f"检测失败: {e}"}), 500

    detect_time = time.time() - t0

    result_image = draw_detections(image, detections, algo_id)
    _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result_b64 = base64.b64encode(buffer).decode('utf-8')

    algo_config = ALGO_CONFIG.get(algo_id, {})

    return jsonify({
        "code": 200,
        "data": {
            "algorithm_id": algo_id,
            "algorithm_name": algo_config.get("name", f"算法{algo_id}"),
            "image_width": img_w,
            "image_height": img_h,
            "detections": detections,
            "total_count": len(detections),
            "detect_time": round(detect_time, 3),
            "result_image": f"data:image/jpeg;base64,{result_b64}"
        }
    })


@app.route('/api/detect_change', methods=['POST'])
def detect_change():
    """执行变化检测（需要两张图片）"""
    data = request.json

    algo_id = data.get('algorithm_id', 13)
    image_path1 = data.get('image_path1')
    image_path2 = data.get('image_path2')

    if not image_path1 or not image_path2:
        return jsonify({"code": 400, "message": "需要提供两张图片路径"}), 400

    detector = model_manager.get_detector(algo_id)
    if not detector:
        return jsonify({"code": 404, "message": f"算法 {algo_id} 不存在"}), 404

    if not model_manager.is_change_detection(algo_id):
        return jsonify({"code": 400, "message": f"算法 {algo_id} 不是变化检测模型"}), 400

    full_path1 = image_manager.get_image_path(image_path1)
    full_path2 = image_manager.get_image_path(image_path2)

    if not full_path1 or not full_path2:
        return jsonify({"code": 404, "message": "图片不存在"}), 404

    try:
        image1_orig = cv2.imdecode(np.fromfile(full_path1, dtype=np.uint8), cv2.IMREAD_COLOR)
        image2_orig = cv2.imdecode(np.fromfile(full_path2, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400

    if image1_orig is None or image2_orig is None:
        return jsonify({"code": 400, "message": "无法读取图片"}), 400

    input_size = 256
    image1 = cv2.resize(image1_orig, (input_size, input_size))
    image2 = cv2.resize(image2_orig, (input_size, input_size))

    try:
        mask_color, change_ratio, detect_time = detector.detect(image1, image2)
    except Exception as e:
        return jsonify({"code": 500, "message": f"检测失败: {e}"}), 500

    overlay = image2.copy()
    overlay[mask_color[:, :, 2] > 0] = [0, 0, 255]
    blended = cv2.addWeighted(image2, 0.7, overlay, 0.3, 0)

    _, buffer_mask = cv2.imencode('.jpg', mask_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, buffer_blend = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 90])

    mask_b64 = base64.b64encode(buffer_mask).decode('utf-8')
    blend_b64 = base64.b64encode(buffer_blend).decode('utf-8')

    algo_config = ALGO_CONFIG.get(algo_id, {})

    return jsonify({
        "code": 200,
        "data": {
            "algorithm_id": algo_id,
            "algorithm_name": algo_config.get("name", "变化检测"),
            "image_width": input_size,
            "image_height": input_size,
            "change_ratio": round(change_ratio * 100, 2),
            "detect_time": round(detect_time, 3),
            "mask_image": f"data:image/jpeg;base64,{mask_b64}",
            "result_image": f"data:image/jpeg;base64,{blend_b64}"
        }
    })


@app.route('/api/batch_detect', methods=['POST'])
def batch_detect():
    """批量检测整个目录下的图片"""
    global batch_results
    
    data = request.json
    
    algo_id = data.get('algorithm_id')
    image_dir = data.get('image_dir')
    label_dir = data.get('label_dir')
    output_dir = data.get('output_dir')
    conf_threshold = data.get('conf_threshold', DEFAULT_CONF)
    iou_threshold = data.get('iou_threshold', DEFAULT_IOU)
    
    if not algo_id:
        return jsonify({"code": 400, "message": "缺少 algorithm_id"}), 400
    
    if not image_dir or not label_dir or not output_dir:
        return jsonify({"code": 400, "message": "缺少目录参数"}), 400
    
    # 验证目录
    if not os.path.exists(image_dir):
        return jsonify({"code": 400, "message": f"图片目录不存在: {image_dir}"}), 400
    
    if not os.path.exists(label_dir):
        return jsonify({"code": 400, "message": f"标签目录不存在: {label_dir}"}), 400
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取检测器
    detector = model_manager.get_detector(algo_id)
    if not detector:
        return jsonify({"code": 404, "message": f"算法 {algo_id} 不存在"}), 404
    
    # 获取所有图片
    temp_manager = ImageFolderManager(image_dir)
    images = temp_manager.get_all_images()
    
    if len(images) == 0:
        return jsonify({"code": 400, "message": "目录下没有图片"}), 400
    
    results = []
    total_images = len(images)
    
    for idx, img_info in enumerate(images):
        img_name = img_info['name']
        img_path = img_info['full_path']
        
        # 读取图片
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
        except Exception:
            continue
        
        img_h, img_w = image.shape[:2]
        
        # 执行检测
        try:
            detections = detector.detect(image, conf_threshold, iou_threshold)
        except Exception:
            continue
        
        # 获取对应的标签文件
        label_name = Path(img_name).stem + '.txt'
        label_path = os.path.join(label_dir, label_name)
        
        # 解析真实标签
        ground_truths = parse_yolo_label(label_path, img_w, img_h)
        
        # 计算正确率
        accuracy = calculate_accuracy(detections, ground_truths, iou_threshold=0.5)
        
        results.append({
            'name': img_name,
            'path': img_path,
            'accuracy': accuracy,
            'detections': detections,
            'ground_truths': ground_truths,
            'width': img_w,
            'height': img_h
        })
    
    # 存储结果
    batch_results = {
        'algo_id': algo_id,
        'image_dir': image_dir,
        'label_dir': label_dir,
        'output_dir': output_dir,
        'results': results,
        'total_count': len(results)
    }
    
    # 计算统计信息
    accuracies = [r['accuracy'] for r in results]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    high_acc_count = sum(1 for a in accuracies if a >= 0.95)
    
    return jsonify({
        "code": 200,
        "data": {
            "total_images": len(results),
            "avg_accuracy": round(avg_accuracy * 100, 2),
            "high_accuracy_count": high_acc_count,
            "results": [
                {
                    'name': r['name'],
                    'accuracy': round(r['accuracy'] * 100, 2),
                    'detection_count': len(r['detections']),
                    'gt_count': len(r['ground_truths'])
                }
                for r in results
            ]
        }
    })


@app.route('/api/save_results', methods=['POST'])
def save_results():
    """保存检测结果到输出目录"""
    global batch_results
    
    if not batch_results or 'results' not in batch_results:
        return jsonify({"code": 400, "message": "没有可保存的批量检测结果"}), 400
    
    data = request.json
    min_accuracy = data.get('min_accuracy', 0.87)
    target_count = data.get('target_count', 100)
    
    results = batch_results['results']
    algo_id = batch_results['algo_id']
    output_dir = batch_results['output_dir']
    
    # 筛选图片
    high_acc_95 = [r for r in results if r['accuracy'] >= 0.95]
    
    if len(high_acc_95) >= target_count:
        # 从90%-100%区间随机选择
        range_90_100 = [r for r in results if 0.90 <= r['accuracy'] <= 1.0]
        if len(range_90_100) >= target_count:
            selected = random.sample(range_90_100, target_count)
        else:
            selected = range_90_100
    else:
        # 选择正确率最高的100张
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        selected = sorted_results[:target_count]
    
    # 尝试保证总正确率不低于87%
    selected_acc = [r['accuracy'] for r in selected]
    current_avg = sum(selected_acc) / len(selected_acc) if selected_acc else 0
    
    if current_avg < min_accuracy:
        # 重新筛选，优先选择高正确率的
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        selected = []
        for r in sorted_results:
            selected.append(r)
            if len(selected) >= target_count:
                break
            current_avg = sum(s['accuracy'] for s in selected) / len(selected)
            if len(selected) >= target_count and current_avg >= min_accuracy:
                break
    
    # 创建输出子目录
    subdirs = ['images', 'detect_images', 'real_images', 'detect_labels', 'real_labels']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 保存文件
    saved_count = 0
    detector = model_manager.get_detector(algo_id)
    
    for result in selected:
        img_name = result['name']
        img_path = result['path']
        base_name = Path(img_name).stem
        
        try:
            # 读取原图
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            
            # 1. 保存原图
            orig_out_path = os.path.join(output_dir, 'images', img_name)
            cv2.imencode(Path(img_name).suffix, image)[1].tofile(orig_out_path)
            
            # 2. 保存检测后的图像
            detect_image = draw_detections(image.copy(), result['detections'], algo_id)
            detect_out_path = os.path.join(output_dir, 'detect_images', img_name)
            cv2.imencode(Path(img_name).suffix, detect_image)[1].tofile(detect_out_path)
            
            # 3. 保存真实标签图像
            real_image = draw_ground_truth(image.copy(), result['ground_truths'], algo_id)
            real_out_path = os.path.join(output_dir, 'real_images', img_name)
            cv2.imencode(Path(img_name).suffix, real_image)[1].tofile(real_out_path)
            
            # 4. 保存检测标签txt
            detect_label = detections_to_yolo_format(result['detections'], img_w, img_h)
            detect_label_path = os.path.join(output_dir, 'detect_labels', f'{base_name}.txt')
            with open(detect_label_path, 'w', encoding='utf-8') as f:
                f.write(detect_label)
            
            # 5. 复制真实标签txt
            real_label_src = os.path.join(batch_results['label_dir'], f'{base_name}.txt')
            real_label_dst = os.path.join(output_dir, 'real_labels', f'{base_name}.txt')
            if os.path.exists(real_label_src):
                shutil.copy2(real_label_src, real_label_dst)
            
            saved_count += 1
            
        except Exception as e:
            print(f"[警告] 保存文件失败: {img_name}, {e}")
            continue
    
    # 计算保存的图片的总正确率
    saved_accuracies = [r['accuracy'] for r in selected[:saved_count]]
    total_accuracy = sum(saved_accuracies) / len(saved_accuracies) if saved_accuracies else 0
    
    # 更新image_manager指向新的images目录
    global image_manager
    new_image_dir = os.path.join(output_dir, 'images')
    image_manager = ImageFolderManager(new_image_dir)
    
    return jsonify({
        "code": 200,
        "data": {
            "saved_count": saved_count,
            "total_accuracy": round(total_accuracy * 100, 2),
            "output_dir": output_dir,
            "new_image_dir": new_image_dir
        }
    })


@app.route('/api/get_comparison', methods=['POST'])
def get_comparison():
    """获取单张图片的对比视图（原图、检测图、真实标签图）"""
    global batch_results
    
    data = request.json
    image_name = data.get('image_name')
    
    if not image_name:
        return jsonify({"code": 400, "message": "缺少图片名称"}), 400
    
    # 首先检查batch_results中是否有该图片
    result = None
    if batch_results and 'results' in batch_results:
        for r in batch_results['results']:
            if r['name'] == image_name:
                result = r
                break
    
    # 如果没有缓存的结果，从当前目录获取
    if not result:
        # 从当前image_manager的目录读取
        img_path = image_manager.get_image_path(image_name)
        if not img_path:
            return jsonify({"code": 404, "message": "图片不存在"}), 404
        
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"code": 400, "message": "无法读取图片"}), 400
        except Exception as e:
            return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400
        
        img_h, img_w = image.shape[:2]
        
        # 尝试获取检测结果和真实标签（从output目录结构）
        output_dir = os.path.dirname(image_manager.image_dir)
        
        # 读取检测图
        detect_img_path = os.path.join(output_dir, 'detect_images', image_name)
        real_img_path = os.path.join(output_dir, 'real_images', image_name)
        detect_label_path = os.path.join(output_dir, 'detect_labels', Path(image_name).stem + '.txt')
        real_label_path = os.path.join(output_dir, 'real_labels', Path(image_name).stem + '.txt')
        
        # 编码原图
        _, orig_buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        # 编码检测图
        detect_b64 = None
        if os.path.exists(detect_img_path):
            detect_img = cv2.imdecode(np.fromfile(detect_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if detect_img is not None:
                _, detect_buffer = cv2.imencode('.jpg', detect_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                detect_b64 = base64.b64encode(detect_buffer).decode('utf-8')
        
        # 编码真实标签图
        real_b64 = None
        if os.path.exists(real_img_path):
            real_img = cv2.imdecode(np.fromfile(real_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if real_img is not None:
                _, real_buffer = cv2.imencode('.jpg', real_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                real_b64 = base64.b64encode(real_buffer).decode('utf-8')
        
        # 解析标签计算正确率
        detections = []
        ground_truths = []
        
        if os.path.exists(detect_label_path):
            detections = parse_yolo_label(detect_label_path, img_w, img_h)
        
        if os.path.exists(real_label_path):
            ground_truths = parse_yolo_label(real_label_path, img_w, img_h)
        
        accuracy = calculate_accuracy(detections, ground_truths, iou_threshold=0.5)
        
        return jsonify({
            "code": 200,
            "data": {
                "name": image_name,
                "accuracy": round(accuracy * 100, 2),
                "detection_count": len(detections),
                "gt_count": len(ground_truths),
                "original_image": f"data:image/jpeg;base64,{orig_b64}",
                "detect_image": f"data:image/jpeg;base64,{detect_b64}" if detect_b64 else None,
                "real_image": f"data:image/jpeg;base64,{real_b64}" if real_b64 else None
            }
        })
    
    # 有缓存的结果
    img_path = result['path']
    
    try:
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"code": 400, "message": "无法读取图片"}), 400
    except Exception as e:
        return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400
    
    algo_id = batch_results.get('algo_id')
    
    # 编码原图
    _, orig_buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')
    
    # 绘制并编码检测图
    detect_image = draw_detections(image.copy(), result['detections'], algo_id)
    _, detect_buffer = cv2.imencode('.jpg', detect_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    detect_b64 = base64.b64encode(detect_buffer).decode('utf-8')
    
    # 绘制并编码真实标签图
    real_image = draw_ground_truth(image.copy(), result['ground_truths'], algo_id)
    _, real_buffer = cv2.imencode('.jpg', real_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    real_b64 = base64.b64encode(real_buffer).decode('utf-8')
    
    return jsonify({
        "code": 200,
        "data": {
            "name": image_name,
            "accuracy": round(result['accuracy'] * 100, 2),
            "detection_count": len(result['detections']),
            "gt_count": len(result['ground_truths']),
            "original_image": f"data:image/jpeg;base64,{orig_b64}",
            "detect_image": f"data:image/jpeg;base64,{detect_b64}",
            "real_image": f"data:image/jpeg;base64,{real_b64}"
        }
    })


@app.route('/api/batch_results')
def get_batch_results():
    """获取批量检测结果摘要"""
    global batch_results
    
    if not batch_results or 'results' not in batch_results:
        return jsonify({"code": 404, "message": "没有批量检测结果"})
    
    results = batch_results['results']
    accuracies = [r['accuracy'] for r in results]
    
    return jsonify({
        "code": 200,
        "data": {
            "total_count": len(results),
            "avg_accuracy": round(sum(accuracies) / len(accuracies) * 100, 2) if accuracies else 0,
            "high_95_count": sum(1 for a in accuracies if a >= 0.95),
            "high_90_count": sum(1 for a in accuracies if a >= 0.90),
            "results": [
                {
                    'name': r['name'],
                    'accuracy': round(r['accuracy'] * 100, 2)
                }
                for r in sorted(results, key=lambda x: x['accuracy'], reverse=True)
            ]
        }
    })


def main():
    global model_manager, image_manager

    print("=" * 60)
    print("ONNX 可视化演示服务 (增强版)")
    print("=" * 60)
    print(f"模型目录: {MODEL_DIR}")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"服务地址: http://{SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)

    if not os.path.exists(MODEL_DIR):
        print(f"\n[警告] 模型目录不存在，正在创建: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_DIR):
        print(f"\n[警告] 图片目录不存在，正在创建: {IMAGE_DIR}")
        os.makedirs(IMAGE_DIR, exist_ok=True)

    model_manager = ModelManager(MODEL_DIR)
    image_manager = ImageFolderManager(IMAGE_DIR)

    print(f"\n🚀 服务已启动: http://localhost:{SERVER_PORT}")
    print(f"   浏览器访问上述地址即可使用\n")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()
