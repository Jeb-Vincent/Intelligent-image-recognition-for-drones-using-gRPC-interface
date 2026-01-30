#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 批量标签生成工具 (多GPU并行) - PyTorch .pt 模型版

功能:
- 自动扫描指定目录下以算法编号开头的文件夹（如01_xxx, 02_xxx...24_xxx）
- 跳过13号变化检测算法
- 对每个文件夹的images子目录进行YOLO检测
- 在images旁边创建labels文件夹，保存YOLO格式的txt标签
- 模型文件（1.pt, 2.pt...24.pt）放在脚本同目录下
- 根据 algo_classes.yaml 配置自动选择模型类型

使用方法:
    1. 安装依赖: pip install ultralytics torch pyyaml opencv-python
    2. 将模型文件 (1.pt, 2.pt...24.pt) 放在脚本同目录下
    3. 将 algo_classes.yaml 放在脚本同目录下
    4. 修改下方 DATA_ROOT 为你的数据目录路径
    5. 运行: python batch_label_generator.py

目录结构示例:
    DATA_ROOT/
    ├── 01_松线虫害识别/
    │   └── images/
    │       ├── img001.jpg
    │       └── img002.jpg
    ├── 02_河道淤积识别/
    │   └── images/
    │       └── ...
    └── ...

输出:
    DATA_ROOT/
    ├── 01_松线虫害识别/
    │   ├── images/
    │   └── labels/          <-- 自动创建
    │       ├── img001.txt
    │       └── img002.txt
    └── ...
"""

import os
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml

# ============================================================
# ★★★ 配置区域 ★★★
# ============================================================

# 数据根目录（包含01_xxx, 02_xxx...等文件夹）
DATA_ROOT = r'D:\演示数据集3\演示数据集'

# 脚本所在目录（模型文件和yaml配置文件所在位置）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置文件路径
YAML_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'algo_classes.yaml')

# GPU配置
NUM_GPUS = 4
GPU_IDS = [0, 1, 2, 3]

# 检测参数
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ============================================================
# 从YAML加载算法配置
# ============================================================
def load_algo_config_from_yaml(yaml_path: str) -> Dict:
    """从yaml文件加载算法配置"""
    if not os.path.exists(yaml_path):
        print(f"[警告] YAML配置文件不存在: {yaml_path}")
        return {}

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        algorithms = config.get('algorithms', {})

        algo_config = {}
        model_types = {}

        for algo_id, algo_info in algorithms.items():
            algo_id = int(algo_id)

            model_type = algo_info.get('model_type', 'yolov5')
            model_types[algo_id] = model_type

            classes = {}
            for cls_id, cls_info in algo_info.get('classes', {}).items():
                cls_id = int(cls_id)
                classes[cls_id] = cls_info.get('name_cn', cls_info.get('name', f'class_{cls_id}'))

            algo_config[algo_id] = {
                'name': algo_info.get('name', f'算法{algo_id}'),
                'classes': classes,
                'num_classes': len(classes)
            }

        return {'algo_config': algo_config, 'model_types': model_types}

    except Exception as e:
        print(f"[错误] 加载YAML配置失败: {e}")
        return {}


# 加载配置
_yaml_config = load_algo_config_from_yaml(YAML_CONFIG_PATH)
ALGO_CONFIG = _yaml_config.get('algo_config', {})
MODEL_TYPES = _yaml_config.get('model_types', {})

# 如果yaml加载失败，使用默认配置
if not MODEL_TYPES:
    print("[警告] YAML加载失败，使用默认模型类型配置")
    MODEL_TYPES = {
        1: "yolov5", 2: "yolov5", 3: "yolov5", 4: "yolov5", 5: "yolov5",
        6: "yolov11", 7: "yolov11_720", 8: "yolov11_720", 9: "yolov5",
        10: "yolov11_720", 11: "yolov5", 12: "yolov11_720", 14: "yolov11_720",
        15: "yolov11_720", 16: "yolov11_720", 17: "yolov5", 18: "yolov11_720",
        19: "yolov5", 20: "yolov5", 21: "yolov5", 22: "yolov5", 23: "yolov5", 24: "yolov5"
    }


# ============================================================
# 工具函数
# ============================================================
def log(msg: str, level: str = "INFO"):
    """打印日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def find_model_path(algo_id: int, model_type: str = None) -> Optional[str]:
    """查找模型文件路径

    支持的命名格式：
    - 基础格式: 1.pt, 01.pt
    - 720后缀: 7_720.pt, 07_720.pt (用于yolov11_720模型)
    - best格式: 1_best.pt, best_1.pt
    - algo格式: algo_1.pt
    """
    # 根据模型类型确定是否需要_720后缀
    is_720 = model_type == 'yolov11_720' if model_type else False

    possible_names = []

    if is_720:
        # 优先查找带_720后缀的模型
        possible_names.extend([
            f'{algo_id}_720.pt',
            f'{algo_id:02d}_720.pt',
            f'{algo_id}_720_best.pt',
            f'best_{algo_id}_720.pt',
            f'algo_{algo_id}_720.pt',
        ])

    # 通用命名格式
    possible_names.extend([
        f'{algo_id}.pt',
        f'{algo_id:02d}.pt',
        f'{algo_id}_best.pt',
        f'{algo_id:02d}_best.pt',
        f'best_{algo_id}.pt',
        f'best_{algo_id:02d}.pt',
        f'algo_{algo_id}.pt',
        f'algo_{algo_id:02d}.pt',
        # 如果不是720类型，也尝试查找_720版本作为备选
        f'{algo_id}_720.pt',
        f'{algo_id:02d}_720.pt',
    ])

    # 去重并保持顺序
    seen = set()
    unique_names = []
    for name in possible_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    for name in unique_names:
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(path):
            return path

    return None


def extract_algo_id_from_folder(folder_name: str) -> Optional[int]:
    """从文件夹名称提取算法编号

    支持的格式:
    - "01_松线虫害识别" -> 1
    - "01" -> 1
    - "1_xxx" -> 1
    - "24_林业侵占" -> 24
    """
    # 匹配开头的数字
    match = re.match(r'^(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def scan_data_folders(data_root: str) -> List[Dict]:
    """扫描数据目录，找出所有算法文件夹"""
    if not os.path.exists(data_root):
        log(f"数据目录不存在: {data_root}", "ERROR")
        return []

    folders = []

    for item in sorted(os.listdir(data_root)):
        item_path = os.path.join(data_root, item)

        if not os.path.isdir(item_path):
            continue

        algo_id = extract_algo_id_from_folder(item)
        if algo_id is None:
            continue

        # 跳过13号变化检测
        if algo_id == 13:
            log(f"跳过算法 13 (变化检测): {item}")
            continue

        # 检查images子目录是否存在
        images_dir = os.path.join(item_path, 'images')
        if not os.path.exists(images_dir):
            log(f"跳过 {item}: images子目录不存在", "WARN")
            continue

        # 检查是否有图片
        image_count = sum(1 for f in os.listdir(images_dir)
                          if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS)
        if image_count == 0:
            log(f"跳过 {item}: images目录中没有图片", "WARN")
            continue

        folders.append({
            'algo_id': algo_id,
            'folder_name': item,
            'folder_path': item_path,
            'images_dir': images_dir,
            'labels_dir': os.path.join(item_path, 'labels'),
            'image_count': image_count
        })

    return folders


def get_all_images(image_dir: str) -> List[Dict]:
    """获取目录下所有图片"""
    if not os.path.exists(image_dir):
        return []

    images = []
    for filename in sorted(os.listdir(image_dir)):
        filepath = os.path.join(image_dir, filename)
        if os.path.isfile(filepath) and Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append({
                'name': filename,
                'full_path': filepath,
                'stem': Path(filename).stem
            })
    return images


def detections_to_yolo_format(detections: List[Dict], img_width: int, img_height: int) -> str:
    """将检测结果转换为YOLO格式"""
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        # 边界检查
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return '\n'.join(lines)


# ============================================================
# YOLOv5 模型加载器
# ============================================================
class YOLOv5Detector:
    """YOLOv5 检测器（使用torch.hub加载）"""

    def __init__(self, model_path: str, device: int = 0, conf: float = 0.25, iou: float = 0.45):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.model = None
        self.names = {}

        self._load_model()

    def _load_model(self):
        """加载YOLOv5模型"""
        import warnings
        warnings.filterwarnings('ignore')

        for force_reload in [False, True]:
            try:
                if force_reload:
                    print(f"    尝试强制重新下载yolov5...")

                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=self.model_path,
                    force_reload=force_reload,
                    trust_repo=True,
                    device=self.device
                )
                self.model.conf = self.conf
                self.model.iou = self.iou

                if hasattr(self.model, 'names'):
                    self.names = self.model.names

                return
            except Exception as e:
                if not force_reload:
                    print(f"    torch.hub加载失败(cached): {e}")
                else:
                    print(f"    torch.hub加载失败(reload): {e}")

        # 备用方法：直接加载
        try:
            checkpoint = torch.load(self.model_path, map_location=f'cuda:{self.device}', weights_only=False)

            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model'].float()
                elif 'ema' in checkpoint and checkpoint['ema'] is not None:
                    self.model = checkpoint['ema'].float()

                if 'names' in checkpoint:
                    self.names = checkpoint['names']
            else:
                self.model = checkpoint.float()

            self.model = self.model.to(f'cuda:{self.device}')
            self.model.eval()

            if hasattr(self.model, 'fuse'):
                try:
                    self.model = self.model.fuse()
                except:
                    pass

        except Exception as e:
            raise RuntimeError(f"所有加载方式都失败: {e}")

    def predict(self, image_or_path, input_size: int = 640) -> List[Dict]:
        """推理"""
        if hasattr(self.model, 'conf'):
            results = self.model(image_or_path, size=input_size)

            detections = []
            pred = results.xyxy[0].cpu().numpy()

            for det in pred:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)

                if isinstance(self.names, dict):
                    class_name = self.names.get(cls_id, f"class_{cls_id}")
                elif isinstance(self.names, list) and cls_id < len(self.names):
                    class_name = self.names[cls_id]
                else:
                    class_name = f"class_{cls_id}"

                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "class_name": class_name
                })

            return detections
        else:
            return self._predict_raw(image_or_path, input_size)

    def _predict_raw(self, image: np.ndarray, input_size: int = 640) -> List[Dict]:
        """原生模型推理"""
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)

        orig_h, orig_w = image.shape[:2]
        img_tensor, r, (pad_left, pad_top) = self._preprocess(image, input_size)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs

        if output.dim() == 3:
            output = output[0]

        output = output.cpu().numpy()

        boxes = output[:, :4]
        objectness = output[:, 4]
        class_scores = output[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        class_conf = np.max(class_scores, axis=1)
        scores = objectness * class_conf

        mask = scores > self.conf
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
        boxes_xyxy /= r

        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes_xyxy) == 0:
            return []

        keep = self._nms(boxes_xyxy, scores, self.iou)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        detections = []
        for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
            cls_id = int(cls_id)
            detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "confidence": float(score),
                "class_id": cls_id,
                "class_name": f"class_{cls_id}"
            })

        return detections

    def _preprocess(self, image: np.ndarray, input_size: int = 640) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """预处理图像"""
        orig_h, orig_w = image.shape[:2]

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
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).to(f'cuda:{self.device}')

        return img_tensor, r, (left, top)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """非极大值抑制"""
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
# 单个算法处理
# ============================================================
def process_single_algorithm(gpu_id: int, task: Dict, conf_threshold: float,
                             iou_threshold: float, script_dir: str,
                             yaml_config_path: str) -> Dict:
    """
    在指定GPU上处理单个算法的所有图片并生成标签
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    algo_id = task['algo_id']
    folder_name = task['folder_name']
    images_dir = task['images_dir']
    labels_dir = task['labels_dir']

    # 重新加载配置（子进程需要）
    _config = load_algo_config_from_yaml(yaml_config_path)
    algo_config = _config.get('algo_config', {})
    model_types = _config.get('model_types', {})

    algo_name = algo_config.get(algo_id, {}).get('name', f'算法{algo_id}')
    algo_classes = algo_config.get(algo_id, {}).get('classes', {})
    model_type = model_types.get(algo_id, 'yolov5')

    log(f"[GPU{gpu_id}] 开始处理算法 {algo_id}: {algo_name}")
    log(f"[GPU{gpu_id}]   文件夹: {folder_name}")
    log(f"[GPU{gpu_id}]   模型类型: {model_type}")

    t_start = time.time()

    # 查找模型
    model_path = find_model_path_in_dir(algo_id, script_dir, model_type)
    if not model_path:
        log(f"[GPU{gpu_id}] 找不到算法 {algo_id} 的模型文件", "ERROR")
        return {'algo_id': algo_id, 'success': False, 'error': '模型文件不存在'}

    log(f"[GPU{gpu_id}]   模型路径: {model_path}")

    # 根据模型类型确定输入尺寸
    if model_type == 'yolov11_720':
        input_size = 736
    elif model_type == 'yolov11':
        input_size = 640
    else:
        input_size = 640

    # 加载模型
    model = None
    loader_type = None

    if model_type in ['yolov11', 'yolov11_720']:
        try:
            from ultralytics import YOLO
            import warnings
            warnings.filterwarnings('ignore')

            model = YOLO(model_path)
            loader_type = 'ultralytics'
            log(f"[GPU{gpu_id}] 模型加载成功 (ultralytics)")

        except Exception as e:
            log(f"[GPU{gpu_id}] ultralytics加载失败: {e}", "ERROR")
            return {'algo_id': algo_id, 'success': False, 'error': str(e)}
    else:
        try:
            model = YOLOv5Detector(model_path, device=0, conf=conf_threshold, iou=iou_threshold)
            loader_type = 'yolov5_custom'
            log(f"[GPU{gpu_id}] 模型加载成功 (YOLOv5)")

        except Exception as e:
            log(f"[GPU{gpu_id}] YOLOv5加载失败: {e}", "ERROR")
            return {'algo_id': algo_id, 'success': False, 'error': str(e)}

    # 创建labels目录
    os.makedirs(labels_dir, exist_ok=True)
    log(f"[GPU{gpu_id}]   输出目录: {labels_dir}")

    # 获取所有图片
    images = get_all_images(images_dir)
    if not images:
        log(f"[GPU{gpu_id}] 算法 {algo_id} 没有找到图片", "WARN")
        return {'algo_id': algo_id, 'success': False, 'error': '没有图片'}

    log(f"[GPU{gpu_id}]   共 {len(images)} 张图片待处理")

    # 处理每张图片
    processed_count = 0
    detected_count = 0
    error_count = 0

    for idx, img_info in enumerate(images):
        img_name = img_info['name']
        img_path = img_info['full_path']
        img_stem = img_info['stem']

        try:
            # 读取图片
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                error_count += 1
                continue

            img_h, img_w = image.shape[:2]

            # 推理
            detections = []

            if loader_type == 'ultralytics':
                yolo_results = model.predict(
                    source=img_path,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=input_size,
                    device=0,
                    verbose=False
                )

                if len(yolo_results) > 0:
                    result = yolo_results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls_id = int(boxes.cls[i].cpu().numpy())

                            detections.append({
                                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                                "confidence": conf,
                                "class_id": cls_id
                            })

            elif loader_type == 'yolov5_custom':
                detections = model.predict(img_path, input_size=input_size)

            # 转换为YOLO格式并保存
            label_content = detections_to_yolo_format(detections, img_w, img_h)
            label_path = os.path.join(labels_dir, f'{img_stem}.txt')

            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(label_content)

            processed_count += 1
            if len(detections) > 0:
                detected_count += 1

            if (idx + 1) % 100 == 0:
                log(f"[GPU{gpu_id}] 算法{algo_id}: 已处理 {idx + 1}/{len(images)}")

        except Exception as e:
            error_count += 1
            continue

    t_end = time.time()
    elapsed = t_end - t_start

    log(f"[GPU{gpu_id}] 算法 {algo_id} 完成!")
    log(f"[GPU{gpu_id}]   处理图片: {processed_count}/{len(images)}")
    log(f"[GPU{gpu_id}]   检测到目标: {detected_count} 张")
    log(f"[GPU{gpu_id}]   错误: {error_count} 张")
    log(f"[GPU{gpu_id}]   耗时: {elapsed:.1f}秒")

    return {
        'algo_id': algo_id,
        'algo_name': algo_name,
        'folder_name': folder_name,
        'success': True,
        'total_images': len(images),
        'processed_count': processed_count,
        'detected_count': detected_count,
        'error_count': error_count,
        'elapsed_time': elapsed
    }


def find_model_path_in_dir(algo_id: int, script_dir: str, model_type: str = None) -> Optional[str]:
    """在指定目录中查找模型文件

    支持的命名格式：
    - 基础格式: 1.pt, 01.pt
    - 720后缀: 7_720.pt, 07_720.pt (用于yolov11_720模型)
    - best格式: 1_best.pt, best_1.pt
    - algo格式: algo_1.pt
    """
    # 根据模型类型确定是否需要_720后缀
    is_720 = model_type == 'yolov11_720' if model_type else False

    possible_names = []

    if is_720:
        # 优先查找带_720后缀的模型
        possible_names.extend([
            f'{algo_id}_720.pt',
            f'{algo_id:02d}_720.pt',
            f'{algo_id}_720_best.pt',
            f'best_{algo_id}_720.pt',
            f'algo_{algo_id}_720.pt',
        ])

    # 通用命名格式
    possible_names.extend([
        f'{algo_id}.pt',
        f'{algo_id:02d}.pt',
        f'{algo_id}_best.pt',
        f'{algo_id:02d}_best.pt',
        f'best_{algo_id}.pt',
        f'best_{algo_id:02d}.pt',
        f'algo_{algo_id}.pt',
        f'algo_{algo_id:02d}.pt',
        # 如果不是720类型，也尝试查找_720版本作为备选
        f'{algo_id}_720.pt',
        f'{algo_id:02d}_720.pt',
    ])

    # 去重并保持顺序
    seen = set()
    unique_names = []
    for name in possible_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    for name in unique_names:
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            return path

    return None


# ============================================================
# 主调度器
# ============================================================
def main():
    """主函数"""
    mp.freeze_support()

    print("=" * 70)
    print("  YOLO 批量标签生成工具 (多GPU并行)")
    print("=" * 70)
    print(f"  数据目录: {DATA_ROOT}")
    print(f"  脚本目录: {SCRIPT_DIR}")
    print(f"  配置文件: {YAML_CONFIG_PATH}")
    print(f"  GPU数量: {NUM_GPUS}")
    print(f"  GPU编号: {GPU_IDS}")
    print(f"  置信度阈值: {DEFAULT_CONF}")
    print(f"  IoU阈值: {DEFAULT_IOU}")
    print("-" * 70)
    print("  支持的模型命名格式:")
    print("    基础: 1.pt, 01.pt, 24.pt")
    print("    720版: 7_720.pt, 07_720.pt (YOLOv11_720模型优先)")
    print("    best: 1_best.pt, best_1.pt, 01_best.pt")
    print("    algo: algo_1.pt, algo_01.pt")
    print("-" * 70)

    print(f"\n  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA设备数: {torch.cuda.device_count()}")
        for i in range(min(torch.cuda.device_count(), NUM_GPUS)):
            print(f"    - GPU{i}: {torch.cuda.get_device_name(i)}")
    print("=" * 70)

    # 扫描数据文件夹
    print("\n扫描数据目录...")
    folders = scan_data_folders(DATA_ROOT)

    if not folders:
        print("\n没有找到可处理的算法文件夹，程序退出。")
        print("请确保数据目录下有以算法编号开头的文件夹（如01_xxx, 02_xxx...）")
        print("且每个文件夹下有images子目录。")
        return

    print(f"\n找到 {len(folders)} 个算法文件夹:")
    for folder in folders:
        model_type = MODEL_TYPES.get(folder['algo_id'], 'yolov5')
        model_path = find_model_path(folder['algo_id'], model_type)
        model_status = "✓" if model_path else "✗"
        print(f"  [{model_status}] 算法 {folder['algo_id']:2d}: {folder['folder_name']} "
              f"({folder['image_count']} 张图片, {model_type})")

    # 检查模型文件
    tasks = []
    skipped = []

    for folder in folders:
        model_type = MODEL_TYPES.get(folder['algo_id'], 'yolov5')
        model_path = find_model_path(folder['algo_id'], model_type)
        if not model_path:
            skipped.append((folder['algo_id'], folder['folder_name'], '模型文件不存在'))
            continue

        tasks.append(folder)

    if skipped:
        print(f"\n跳过 {len(skipped)} 个文件夹 (模型不存在):")
        for algo_id, folder_name, reason in skipped:
            print(f"  - 算法 {algo_id}: {folder_name} ({reason})")

    if not tasks:
        print("\n没有可处理的任务，程序退出。")
        print("请确保脚本目录下有对应的模型文件 (如 1.pt, 2.pt...)")
        return

    print(f"\n待处理: {len(tasks)} 个算法")

    # 确认执行
    print("\n" + "=" * 70)
    user_input = input("  是否开始处理? (y/n): ").strip().lower()
    if user_input != 'y':
        print("  用户取消，程序退出。")
        return

    print("=" * 70 + "\n")
    print("开始处理...\n")

    t_total_start = time.time()

    all_results = []
    task_queue = list(tasks)

    # 使用spawn方式创建进程
    ctx = mp.get_context('spawn')

    # 动态调整GPU数量
    actual_gpus = min(NUM_GPUS, torch.cuda.device_count() if torch.cuda.is_available() else 1, len(tasks))
    actual_gpu_ids = GPU_IDS[:actual_gpus]

    with ProcessPoolExecutor(max_workers=actual_gpus, mp_context=ctx) as executor:
        futures = {}
        gpu_available = list(actual_gpu_ids)

        # 提交初始任务
        while gpu_available and task_queue:
            gpu_id = gpu_available.pop(0)
            task = task_queue.pop(0)

            future = executor.submit(
                process_single_algorithm,
                gpu_id,
                task,
                DEFAULT_CONF,
                DEFAULT_IOU,
                SCRIPT_DIR,
                YAML_CONFIG_PATH
            )
            futures[future] = (gpu_id, task)
            log(f"提交任务: 算法{task['algo_id']} -> GPU{gpu_id}")

        # 等待任务完成并提交新任务
        while futures:
            done_futures = []
            for future in list(futures.keys()):
                if future.done():
                    done_futures.append(future)

            if not done_futures:
                time.sleep(0.5)
                continue

            for future in done_futures:
                gpu_id, task = futures.pop(future)

                try:
                    result = future.result(timeout=10)
                    all_results.append(result)
                except Exception as e:
                    log(f"任务失败: 算法{task['algo_id']} - {e}", "ERROR")
                    all_results.append({
                        'algo_id': task['algo_id'],
                        'folder_name': task['folder_name'],
                        'success': False,
                        'error': str(e)
                    })

                # 提交下一个任务
                if task_queue:
                    next_task = task_queue.pop(0)
                    next_future = executor.submit(
                        process_single_algorithm,
                        gpu_id,
                        next_task,
                        DEFAULT_CONF,
                        DEFAULT_IOU,
                        SCRIPT_DIR,
                        YAML_CONFIG_PATH
                    )
                    futures[next_future] = (gpu_id, next_task)
                    log(f"提交任务: 算法{next_task['algo_id']} -> GPU{gpu_id}")

    t_total_end = time.time()
    total_elapsed = t_total_end - t_total_start

    # 打印汇总报告
    print("\n" + "=" * 70)
    print("  处理完成 - 汇总报告")
    print("=" * 70)

    success_count = sum(1 for r in all_results if r.get('success', False))
    fail_count = len(all_results) - success_count

    print(f"\n总耗时: {total_elapsed:.1f} 秒 ({total_elapsed / 60:.1f} 分钟)")
    print(f"成功: {success_count} 个, 失败: {fail_count} 个")

    print("\n详细结果:")
    print("-" * 100)
    print(f"{'算法ID':<8} {'文件夹':<30} {'总图片':<10} {'已处理':<10} {'有检测':<10} {'耗时(秒)':<10} {'状态':<8}")
    print("-" * 100)

    for r in sorted(all_results, key=lambda x: x.get('algo_id', 0)):
        if r.get('success', False):
            print(f"{r['algo_id']:<8} {r.get('folder_name', '-')[:28]:<30} "
                  f"{r.get('total_images', 0):<10} {r.get('processed_count', 0):<10} "
                  f"{r.get('detected_count', 0):<10} {r.get('elapsed_time', 0):<10.1f} {'✓ 成功':<8}")
        else:
            print(f"{r.get('algo_id', '?'):<8} {r.get('folder_name', '-')[:28]:<30} "
                  f"{'-':<10} {'-':<10} {'-':<10} {'-':<10} {'✗ 失败':<8}")

    print("-" * 100)

    # 保存报告
    report_path = os.path.join(SCRIPT_DIR, f'label_gen_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO 批量标签生成报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {DATA_ROOT}\n")
            f.write(f"总耗时: {total_elapsed:.1f} 秒\n")
            f.write(f"成功: {success_count} 个, 失败: {fail_count} 个\n\n")

            for r in sorted(all_results, key=lambda x: x.get('algo_id', 0)):
                if r.get('success', False):
                    f.write(f"算法 {r['algo_id']}: {r.get('algo_name', '-')}\n")
                    f.write(f"  - 文件夹: {r.get('folder_name', '-')}\n")
                    f.write(f"  - 总图片: {r.get('total_images', 0)}\n")
                    f.write(f"  - 已处理: {r.get('processed_count', 0)}\n")
                    f.write(f"  - 有检测: {r.get('detected_count', 0)}\n")
                    f.write(f"  - 耗时: {r.get('elapsed_time', 0):.1f}秒\n\n")
                else:
                    f.write(f"算法 {r.get('algo_id', '?')}: 失败 - {r.get('error', '未知')}\n\n")

        print(f"\n报告已保存: {report_path}")
    except Exception as e:
        print(f"\n保存报告失败: {e}")

    print("\n所有任务已完成！")
    print(f"标签文件已保存到各算法文件夹的 labels 子目录中。")


if __name__ == '__main__':
    main()