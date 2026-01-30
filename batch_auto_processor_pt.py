#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 批量自动化处理工具 (4×GPU并行) - PyTorch .pt 模型版 V2

功能:
- 同时使用4张GPU (GPU0-3) 并行处理不同算法
- 每处理完一个算法自动添加下一个
- 自动筛选高正确率图片并保存
- 支持24个算法(除13号变化检测外)
- 根据 algo_classes.yaml 配置自动选择模型加载方式
- 支持 YOLOv5 和 YOLOv11/v8 (ultralytics) 模型

使用方法:
    1. 安装依赖: pip install ultralytics torch pyyaml
    2. 将 algo_classes.yaml 放在模型目录下
    3. 在下方 ALGO_PATHS 配置每个算法的目录路径
    4. 运行: python batch_auto_processor_pt.py

注意:
    - YOLOv5模型首次加载会通过torch.hub自动下载yolov5代码，需要网络连接
    - 下载的代码缓存在 ~/.cache/torch/hub/ultralytics_yolov5_master
    - 如果网络不好，可以手动克隆: git clone https://github.com/ultralytics/yolov5
      然后将yolov5文件夹放在模型目录下
"""

import os
import sys
import time
import shutil
import random
import queue
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml

# ============================================================
# ★★★ 配置区域 - 在此处配置每个算法的目录路径 ★★★
# ============================================================

# 模型目录（存放所有.pt模型文件和algo_classes.yaml）
MODEL_DIR = r'./'

# 配置文件路径
YAML_CONFIG_PATH = r'./algo_classes.yaml'

# GPU配置
NUM_GPUS = 4
GPU_IDS = [0, 1, 2, 3]

# 检测参数
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# 筛选参数
TARGET_COUNT = 100        # 目标保存图片数量
MIN_ACCURACY = 0.90       # 最低总正确率要求
MAX_ACCURACY = 0.98       # 最高总正确率限制（避免过高）
TARGET_ACC_MIN = 0.90     # 目标正确率下限
TARGET_ACC_MAX = 0.98     # 目标正确率上限
MAX_NEGATIVE_RATIO = 0.1  # 负样本最大比例（0.1表示最多10%负样本）

# ============================================================
# ★★★ 算法路径配置 - 请修改为您的实际路径 ★★★
# ============================================================

ALGO_PATHS = {
    1: {
        'name': '松线虫害识别',
        'image_dir': r'E:\Dead_tree\deadtree\images\train',
        'label_dir': r'E:\Dead_tree\deadtree\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\1'
    },
    2: {
        'name': '河道淤积识别',
        'image_dir': r'E:\River_sedimentation_identification\images\train',
        'label_dir': r'E:\River_sedimentation_identification\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\2'
    },
    3: {
        'name': '漂浮物识别',
        'image_dir': r'E:\Floating_objects\images\train',
        'label_dir': r'E:\Floating_objects\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\3'
    },
    4: {
        'name': '游泳涉水识别',
        'image_dir': r'E:\Swimming_detection\images\train',
        'label_dir': r'E:\Swimming_detection\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\4'
    },
    5: {
        'name': '车牌识别',
        'image_dir': r'E:\License_plate_recognition\images\train',
        'label_dir': r'E:\License_plate_recognition\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\5'
    },
    6: {
        'name': '交通拥堵识别',
        'image_dir': r'D:\Wenshuo\UAV\6\Vehicle_countingV1.0\images\train',
        'label_dir': r'D:\Wenshuo\UAV\6\Vehicle_countingV1.0\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\6'
    },
    7: {
        'name': '路面破损识别',
        'image_dir': r'D:\Wenshuo\tmp\7\images',
        'label_dir': r'D:\Wenshuo\tmp\7\labels',
        'output_dir': r'D:\Wenshuo\UAVData\7'
    },
    8: {
        'name': '路面污染',
        'image_dir': r'D:\Wenshuo\UAV\8\Road_pollutionV2.0\images\train',
        'label_dir': r'D:\Wenshuo\UAV\8\Road_pollutionV2.0\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\8'
    },
    9: {
        'name': '人群聚集识别',
        'image_dir': r'E:\People_counting\images\train',
        'label_dir': r'E:\People_counting\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\9'
    },
    10: {
        'name': '非法垂钓识别',
        'image_dir': r'D:\Wenshuo\UAV\10\Fishing_detection2\images\train',
        'label_dir': r'D:\Wenshuo\UAV\10\Fishing_detection2\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\10'
    },
    11: {
        'name': '施工识别',
        'image_dir': r'E:\Construction_Equipment\images\train',
        'label_dir': r'E:\Construction_Equipment\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\11'
    },
    12: {
        'name': '秸秆焚烧',
        'image_dir': r'D:\Wenshuo\UAV\12\Straw_burningV1.0\images\train',
        'label_dir': r'D:\Wenshuo\UAV\12\Straw_burningV1.0\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\12'
    },
    # 13号是变化检测，跳过
    14: {
        'name': '占道经营识别',
        'image_dir': r'D:\Wenshuo\UAV\14\Roadside_business\images\train',
        'label_dir': r'D:\Wenshuo\UAV\14\Roadside_business\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\14'
    },
    15: {
        'name': '垃圾堆放识别',
        'image_dir': r'D:\Wenshuo\UAV\15\Trash_detectionV1.0\images\train',
        'label_dir': r'D:\Wenshuo\UAV\15\Trash_detectionV1.0\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\15'
    },
    16: {
        'name': '裸土未覆盖识别',
        'image_dir': r'D:\Wenshuo\UAV\16\Baresoil_detectionV1.0\images\train',
        'label_dir': r'D:\Wenshuo\UAV\16\Baresoil_detectionV1.0\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\16'
    },
    17: {
        'name': '建控区违建识别',
        'image_dir': r'E:\illegal_building\images\train',
        'label_dir': r'E:\illegal_building\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\17'
    },
    18: {
        'name': '烟火识别',
        'image_dir': r'D:\Wenshuo\UAV\18\Fire_DetectionV1.0\images\train',
        'label_dir': r'D:\Wenshuo\UAV\18\Fire_DetectionV1.0\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\18'
    },
    19: {
        'name': '光伏板缺陷检测',
        'image_dir': r'E:\PV_detection\images\train',
        'label_dir': r'E:\PV_detection\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\19'
    },
    20: {
        'name': '园区夜间入侵检测',
        'image_dir': r'E:\tmp\person_car\images',
        'label_dir': r'E:\tmp\person_car\labels',
        'output_dir': r'D:\Wenshuo\UAVData\20'
    },
    21: {
        'name': '园区外立面病害识别',
        'image_dir': r'E:\Wall_defect_identification\images\train',
        'label_dir': r'E:\Wall_defect_identification\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\21'
    },
    22: {
        'name': '罂粟识别',
        'image_dir': r'E:\poppy_detection\images\train',
        'label_dir': r'E:\poppy_detection\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\22'
    },
    23: {
        'name': '作物倒伏检测',
        'image_dir': r'E:\Crop_LodgingV1.1\images\train',
        'label_dir': r'E:\Crop_LodgingV1.1\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\23'
    },
    24: {
        'name': '林业侵占',
        'image_dir': r'E:\Forestry_encroachment\images\train',
        'label_dir': r'E:\Forestry_encroachment\labels\train',
        'output_dir': r'D:\Wenshuo\UAVData\24'
    },
}

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
        
        # 转换格式
        algo_config = {}
        model_types = {}
        
        for algo_id, algo_info in algorithms.items():
            algo_id = int(algo_id)
            
            # 获取模型类型
            model_type = algo_info.get('model_type', 'yolov5')
            model_types[algo_id] = model_type
            
            # 获取类别
            classes = {}
            for cls_id, cls_info in algo_info.get('classes', {}).items():
                cls_id = int(cls_id)
                classes[cls_id] = cls_info.get('name_cn', cls_info.get('name', f'class_{cls_id}'))
            
            algo_config[algo_id] = {
                'name': algo_info.get('name', f'算法{algo_id}'),
                'classes': classes
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
if not ALGO_CONFIG:
    print("[警告] 使用默认算法配置")
    ALGO_CONFIG = {
        1: {"name": "松线虫害识别", "classes": {0: "死亡", 1: "重度患病", 2: "轻度患病"}},
        2: {"name": "河道淤积识别", "classes": {0: "水污染", 1: "漂浮碎片", 2: "废弃船只", 3: "渔业和水产养殖", 4: "垃圾"}},
        3: {"name": "漂浮物识别", "classes": {0: "瓶子", 1: "草", 2: "树枝", 3: "牛奶盒", 4: "塑料袋", 5: "塑料垃圾袋", 6: "球", 7: "叶子"}},
        4: {"name": "游泳涉水识别", "classes": {0: "忽略", 1: "游泳者", 2: "船", 3: "水上摩托艇", 4: "救生设备", 5: "浮标"}},
        5: {"name": "车牌识别", "classes": {0: "车牌"}},
        6: {"name": "交通拥堵识别", "classes": {0: "车辆"}},
        7: {"name": "路面破损识别", "classes": {0: "龟裂", 1: "纵向裂缝", 2: "纵向修补块", 3: "检查井井盖", 4: "坑洞", 5: "横向裂缝", 6: "横向修补块"}},
        8: {"name": "路面污染", "classes": {0: "裂缝", 1: "积水", 2: "路面松散", 3: "泥泞道路", 4: "路边垃圾", 5: "坑洞"}},
        9: {"name": "人群聚集识别", "classes": {0: "车", 1: "人"}},
        10: {"name": "非法垂钓识别", "classes": {0: "水边钓鱼", 1: "游泳溺水", 2: "钓鱼伞", 3: "船"}},
        11: {"name": "施工识别", "classes": {0: "起重机", 1: "挖掘机", 2: "拖拉机", 3: "卡车"}},
        12: {"name": "秸秆焚烧", "classes": {0: "秸秆堆"}},
        14: {"name": "占道经营识别", "classes": {0: "占道经营"}},
        15: {"name": "垃圾堆放识别", "classes": {0: "长椅", 1: "商业垃圾", 2: "非法倾倒点", 3: "绿地", 4: "孔洞", 5: "泽西护栏", 6: "地块", 7: "原材料", 8: "生活垃圾"}},
        16: {"name": "裸土未覆盖识别", "classes": {0: "垃圾", 1: "裸土"}},
        17: {"name": "建控区违建识别", "classes": {0: "蓝色天篷", 1: "其他违建", 2: "改装绿色小屋"}},
        18: {"name": "烟火识别", "classes": {0: "烟雾", 1: "火"}},
        19: {"name": "光伏板缺陷检测", "classes": {0: "有缺陷的光伏电池"}},
        20: {"name": "园区夜间入侵检测", "classes": {0: "人", 1: "车", 2: "自行车"}},
        21: {"name": "园区外立面病害识别", "classes": {0: "墙体腐蚀", 1: "墙体开裂", 2: "墙体劣化", 3: "墙模", 4: "墙面污渍"}},
        22: {"name": "罂粟识别", "classes": {0: "罂粟"}},
        23: {"name": "作物倒伏检测", "classes": {0: "作物倒伏"}},
        24: {"name": "林业侵占", "classes": {0: "反铲装载机", 1: "压路机", 2: "混凝土搅拌车", 3: "推土机", 4: "倾卸卡车", 5: "挖掘机", 6: "平地机", 7: "安全头盔", 8: "移动式起重机", 9: "人", 10: "塔式起重机", 11: "背心", 12: "轮式装载机"}},
    }
    MODEL_TYPES = {
        1: "yolov5", 2: "yolov5", 3: "yolov5", 4: "yolov5", 5: "yolov5",
        6: "yolov11", 7: "yolov11_720", 8: "yolov11_720", 9: "yolov5",
        10: "yolov11_720", 11: "yolov5", 12: "yolov11_720", 14: "yolov11_720",
        15: "yolov11_720", 16: "yolov11_720", 17: "yolov5", 18: "yolov11_720",
        19: "yolov5", 20: "yolov5", 21: "yolov5", 22: "yolov5", 23: "yolov5", 24: "yolov5"
    }

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ============================================================
# 工具函数
# ============================================================
def log(msg: str, level: str = "INFO"):
    """打印日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


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
    
    return inter_area / union_area if union_area > 0 else 0


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
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                    })
    except Exception as e:
        pass
    
    return labels


def calculate_accuracy(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
    """计算检测正确率"""
    if len(ground_truths) == 0 and len(predictions) == 0:
        return 1.0
    if len(ground_truths) == 0 or len(predictions) == 0:
        return 0.0
    
    matched_gt = set()
    matched_pred = set()
    
    matches = []
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            if pred.get('class_id', -1) == gt.get('class_id', -2):
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold:
                    matches.append((iou, i, j))
    
    matches.sort(reverse=True)
    
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    correct_count = len(matched_gt)
    total = max(len(predictions), len(ground_truths))
    
    return correct_count / total if total > 0 else 0.0


def detections_to_yolo_format(detections: List[Dict], img_width: int, img_height: int) -> str:
    """将检测结果转换为YOLO格式"""
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return '\n'.join(lines)


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
                'path': filename,
                'full_path': filepath
            })
    return images


def draw_detections(image: np.ndarray, detections: List[Dict], algo_id: int) -> np.ndarray:
    """在图片上绘制检测结果（绿色框）"""
    from PIL import Image, ImageDraw, ImageFont
    
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_image)
    
    font = None
    font_size = 20
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
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
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_ground_truth(image: np.ndarray, labels: List[Dict], algo_id: int) -> np.ndarray:
    """绘制真实标签（蓝色框）"""
    from PIL import Image, ImageDraw, ImageFont
    
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_image)
    
    font = None
    font_size = 20
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()
    
    colors = [(0, 128, 255), (0, 200, 255), (50, 150, 255), (100, 100, 255)]
    algo_classes = ALGO_CONFIG.get(algo_id, {}).get("classes", {})
    
    for label in labels:
        x1, y1, x2, y2 = map(int, label["bbox"])
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
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ============================================================
# YOLOv5 模型加载器 (用于旧版模型)
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
        
        # 方法1: 使用 torch.hub 加载 (推荐，会自动下载yolov5代码)
        # 先尝试使用缓存，如果失败则强制重新下载
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
        
        # 方法2: 直接加载 (需要yolov5在python路径中)
        try:
            # 尝试添加可能的yolov5路径
            import sys
            possible_paths = [
                os.path.join(os.path.dirname(self.model_path), 'yolov5'),
                os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master'),
                './yolov5',
            ]
            for p in possible_paths:
                if os.path.exists(p) and p not in sys.path:
                    sys.path.insert(0, p)
            
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
        # 如果model是torch.hub加载的，直接用它的推理方法
        if hasattr(self.model, 'conf'):
            # torch.hub 模型
            results = self.model(image_or_path, size=input_size)
            
            detections = []
            pred = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            
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
            # 原生torch模型，需要手动预处理和后处理
            return self._predict_raw(image_or_path, input_size)
    
    def _predict_raw(self, image: np.ndarray, input_size: int = 640) -> List[Dict]:
        """原生模型推理（手动前后处理）"""
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        img_tensor, r, (pad_left, pad_top) = self._preprocess(image, input_size)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # 处理输出
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        
        if output.dim() == 3:
            output = output[0]
        
        output = output.cpu().numpy()
        
        # YOLOv5 输出格式: [num_boxes, 5 + num_classes]
        boxes = output[:, :4]
        objectness = output[:, 4]
        class_scores = output[:, 5:]
        
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = np.max(class_scores, axis=1)
        scores = objectness * class_conf
        
        # 过滤
        mask = scores > self.conf
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # xywh -> xyxy
        boxes_xyxy = np.copy(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # 还原坐标
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= r
        
        # 有效性检查
        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        
        if len(boxes_xyxy) == 0:
            return []
        
        # NMS
        keep = self._nms(boxes_xyxy, scores, self.iou)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        
        # 裁剪
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        
        # 构建结果
        detections = []
        for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
            cls_id = int(cls_id)
            if isinstance(self.names, dict):
                class_name = self.names.get(cls_id, f"class_{cls_id}")
            elif isinstance(self.names, list) and cls_id < len(self.names):
                class_name = self.names[cls_id]
            else:
                class_name = f"class_{cls_id}"
            
            detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "confidence": float(score),
                "class_id": cls_id,
                "class_name": class_name
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
# 单个算法处理进程
# ============================================================
def process_single_algorithm(gpu_id: int, algo_id: int, model_path: str, model_type: str,
                             image_dir: str, label_dir: str, output_dir: str,
                             conf_threshold: float, iou_threshold: float,
                             algo_config: Dict, yaml_config_path: str) -> Dict:
    """
    在指定GPU上处理单个算法的所有图片
    根据 model_type 选择不同的加载方式
    """
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    algo_name = algo_config.get(algo_id, {}).get('name', f'算法{algo_id}')
    algo_classes = algo_config.get(algo_id, {}).get('classes', {})
    
    log(f"[GPU{gpu_id}] 开始处理算法 {algo_id}: {algo_name} (模型类型: {model_type})")
    
    t_start = time.time()
    
    # 根据模型类型确定输入尺寸
    if model_type == 'yolov11_720':
        input_size = 736
    elif model_type == 'yolov11':
        input_size = 640
    else:  # yolov5
        input_size = 640
    
    # 加载模型
    model = None
    loader_type = None
    
    if model_type in ['yolov11', 'yolov11_720']:
        # 使用 ultralytics 加载
        try:
            from ultralytics import YOLO
            import warnings
            warnings.filterwarnings('ignore')
            
            model = YOLO(model_path)
            loader_type = 'ultralytics'
            log(f"[GPU{gpu_id}] 模型加载成功 (ultralytics): {model_path}")
            
        except Exception as e:
            log(f"[GPU{gpu_id}] ultralytics加载失败: {e}", "ERROR")
            return {'algo_id': algo_id, 'success': False, 'error': str(e)}
    
    else:  # yolov5
        # 使用自定义 YOLOv5Detector 加载
        try:
            model = YOLOv5Detector(model_path, device=0, conf=conf_threshold, iou=iou_threshold)
            loader_type = 'yolov5_custom'
            log(f"[GPU{gpu_id}] 模型加载成功 (YOLOv5): {model_path}")
            
        except Exception as e:
            log(f"[GPU{gpu_id}] YOLOv5加载失败: {e}", "ERROR")
            return {'algo_id': algo_id, 'success': False, 'error': str(e)}
    
    # 获取所有图片
    images = get_all_images(image_dir)
    if not images:
        log(f"[GPU{gpu_id}] 算法 {algo_id} 没有找到图片", "WARN")
        return {'algo_id': algo_id, 'success': False, 'error': '没有图片'}
    
    log(f"[GPU{gpu_id}] 算法 {algo_id} 共 {len(images)} 张图片")
    
    # 处理每张图片
    results = []
    
    for idx, img_info in enumerate(images):
        img_name = img_info['name']
        img_path = img_info['full_path']
        
        try:
            # 读取图片
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
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
                            
                            class_name = algo_classes.get(cls_id, f"class_{cls_id}")
                            
                            detections.append({
                                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                                "confidence": conf,
                                "class_id": cls_id,
                                "class_name": class_name
                            })
            
            elif loader_type == 'yolov5_custom':
                detections = model.predict(img_path, input_size=input_size)
                # 更新类别名称
                for det in detections:
                    det['class_name'] = algo_classes.get(det['class_id'], det['class_name'])
            
            # 获取真实标签
            label_name = Path(img_name).stem + '.txt'
            label_path = os.path.join(label_dir, label_name)
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
            
            if (idx + 1) % 100 == 0:
                log(f"[GPU{gpu_id}] 算法{algo_id}: 已处理 {idx + 1}/{len(images)}")
                
        except Exception as e:
            continue
    
    # 筛选并保存结果
    if results:
        saved_count, total_accuracy = save_selected_results(
            results, algo_id, label_dir, output_dir,
            TARGET_COUNT, MIN_ACCURACY, MAX_ACCURACY, MAX_NEGATIVE_RATIO,
            algo_config
        )
    else:
        saved_count, total_accuracy = 0, 0
    
    t_end = time.time()
    elapsed = t_end - t_start
    
    # 统计信息
    accuracies = [r['accuracy'] for r in results]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    high_95_count = sum(1 for a in accuracies if a >= 0.95)
    total_positive = sum(1 for r in results if len(r['ground_truths']) > 0)
    total_negative = len(results) - total_positive
    
    log(f"[GPU{gpu_id}] 算法 {algo_id} 完成!")
    log(f"         处理图片: {len(results)} (正样本:{total_positive}, 负样本:{total_negative})")
    log(f"         平均正确率: {avg_accuracy*100:.1f}%, ≥95%数量: {high_95_count}")
    log(f"         保存数量: {saved_count}, 保存正确率: {total_accuracy*100:.1f}%")
    log(f"         耗时: {elapsed:.1f}秒")
    
    return {
        'algo_id': algo_id,
        'algo_name': algo_name,
        'gpu_id': gpu_id,
        'success': True,
        'total_images': len(results),
        'total_positive': total_positive,
        'total_negative': total_negative,
        'avg_accuracy': avg_accuracy,
        'high_95_count': high_95_count,
        'saved_count': saved_count,
        'saved_accuracy': total_accuracy,
        'elapsed_time': elapsed
    }


def save_selected_results(results: List[Dict], algo_id: int, label_dir: str, output_dir: str,
                          target_count: int, min_accuracy: float, max_accuracy: float,
                          max_negative_ratio: float, algo_config: Dict) -> Tuple[int, float]:
    """
    筛选并保存结果
    策略：将最终正确率控制在 min_accuracy(90%) 到 max_accuracy(98%) 之间
    如果样本正确率普遍较高，会混入一些较低正确率的样本来平衡
    """
    
    positive_samples = [r for r in results if len(r['ground_truths']) > 0]
    negative_samples = [r for r in results if len(r['ground_truths']) == 0]
    
    log(f"  算法{algo_id}: 正样本 {len(positive_samples)} 张, 负样本 {len(negative_samples)} 张")
    
    max_negative_count = int(target_count * max_negative_ratio)
    min_positive_count = target_count - max_negative_count
    
    # 将正样本按正确率分组
    # 高正确率组: >= 98%
    pos_very_high = [r for r in positive_samples if r['accuracy'] >= 0.98]
    # 目标高组: 95%-98%
    pos_high = [r for r in positive_samples if 0.95 <= r['accuracy'] < 0.98]
    # 目标中组: 90%-95%  
    pos_mid = [r for r in positive_samples if 0.90 <= r['accuracy'] < 0.95]
    # 目标低组: 85%-90%
    pos_low = [r for r in positive_samples if 0.85 <= r['accuracy'] < 0.90]
    # 较低组: 80%-85%
    pos_lower = [r for r in positive_samples if 0.80 <= r['accuracy'] < 0.85]
    # 很低组: < 80%
    pos_very_low = [r for r in positive_samples if r['accuracy'] < 0.80]
    
    log(f"  算法{algo_id}: 正确率分布 - >=98%:{len(pos_very_high)}, 95-98%:{len(pos_high)}, "
        f"90-95%:{len(pos_mid)}, 85-90%:{len(pos_low)}, 80-85%:{len(pos_lower)}, <80%:{len(pos_very_low)}")
    
    selected = []
    target_avg = (min_accuracy + max_accuracy) / 2  # 目标平均正确率 ~94%
    
    def calculate_avg_acc(samples):
        """计算样本列表的平均正确率"""
        if not samples:
            return 0
        return sum(r['accuracy'] for r in samples) / len(samples)
    
    def select_with_target_accuracy(candidates, current_selected, target_total, target_min_acc, target_max_acc):
        """
        从候选样本中选择，使最终正确率接近目标范围
        """
        if not candidates:
            return current_selected
        
        remaining_count = target_total - len(current_selected)
        if remaining_count <= 0:
            return current_selected
        
        # 按正确率排序
        sorted_candidates = sorted(candidates, key=lambda x: x['accuracy'], reverse=True)
        
        # 当前平均正确率
        if current_selected:
            current_avg = calculate_avg_acc(current_selected)
            current_sum = sum(r['accuracy'] for r in current_selected)
        else:
            current_avg = 0
            current_sum = 0
        
        new_selected = list(current_selected)
        
        for candidate in sorted_candidates:
            if len(new_selected) >= target_total:
                break
            
            # 模拟添加后的平均正确率
            new_sum = current_sum + candidate['accuracy']
            new_count = len(new_selected) + 1
            new_avg = new_sum / new_count
            
            # 如果添加后正确率仍在合理范围，或者数量不足，就添加
            if new_avg >= target_min_acc - 0.02 or len(new_selected) < target_total * 0.5:
                new_selected.append(candidate)
                current_sum = new_sum
        
        return new_selected
    
    # 策略：混合选择以达到目标正确率范围
    
    # 第一步：优先选择目标范围内的样本 (90%-98%)
    target_range_samples = pos_mid + pos_high  # 90%-98%
    random.shuffle(target_range_samples)
    
    if len(target_range_samples) >= target_count:
        # 目标范围内样本足够，直接随机选择
        selected = random.sample(target_range_samples, target_count)
    else:
        # 先把目标范围内的都选上
        selected = list(target_range_samples)
        
        # 计算当前平均正确率
        current_avg = calculate_avg_acc(selected) if selected else 0
        remaining = target_count - len(selected)
        
        if remaining > 0:
            # 需要补充样本
            # 如果当前正确率偏高，优先补充较低正确率的样本
            # 如果当前正确率偏低，优先补充较高正确率的样本
            
            if current_avg > target_avg:
                # 正确率偏高，需要压低，优先选择较低正确率的
                supplement_order = pos_low + pos_lower + pos_very_low + pos_very_high
            else:
                # 正确率偏低或适中，优先选择较高正确率的
                supplement_order = pos_very_high + pos_low + pos_lower + pos_very_low
            
            # 过滤掉已选择的
            supplement_order = [r for r in supplement_order if r not in selected]
            
            # 补充样本
            for sample in supplement_order:
                if len(selected) >= target_count:
                    break
                selected.append(sample)
    
    # 第二步：检查并调整正确率
    if selected:
        current_avg = calculate_avg_acc(selected)
        
        # 如果正确率过高 (>98%)，需要替换一些高正确率样本为低正确率样本
        if current_avg > max_accuracy:
            log(f"  算法{algo_id}: 当前正确率 {current_avg*100:.1f}% > {max_accuracy*100:.1f}%, 需要压低")
            
            # 按正确率从高到低排序
            selected.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # 找出可替换的低正确率样本
            available_low = [r for r in (pos_low + pos_lower + pos_very_low) if r not in selected]
            available_low.sort(key=lambda x: x['accuracy'])  # 从低到高
            
            # 逐步替换，直到正确率降到目标范围
            replace_idx = 0
            low_idx = 0
            
            while current_avg > max_accuracy and replace_idx < len(selected) and low_idx < len(available_low):
                # 计算替换后的正确率
                old_sample = selected[replace_idx]
                new_sample = available_low[low_idx]
                
                # 只替换正确率>=98%的样本
                if old_sample['accuracy'] >= 0.98:
                    test_avg = (sum(r['accuracy'] for r in selected) - old_sample['accuracy'] + new_sample['accuracy']) / len(selected)
                    
                    if test_avg >= min_accuracy:  # 确保不会降太低
                        selected[replace_idx] = new_sample
                        current_avg = calculate_avg_acc(selected)
                        low_idx += 1
                
                replace_idx += 1
            
            log(f"  算法{algo_id}: 调整后正确率 {current_avg*100:.1f}%")
        
        # 如果正确率过低 (<90%)，需要替换一些低正确率样本为高正确率样本
        elif current_avg < min_accuracy:
            log(f"  算法{algo_id}: 当前正确率 {current_avg*100:.1f}% < {min_accuracy*100:.1f}%, 需要提高")
            
            # 按正确率从低到高排序
            selected.sort(key=lambda x: x['accuracy'])
            
            # 找出可替换的高正确率样本
            available_high = [r for r in (pos_very_high + pos_high) if r not in selected]
            available_high.sort(key=lambda x: x['accuracy'], reverse=True)  # 从高到低
            
            # 逐步替换
            replace_idx = 0
            high_idx = 0
            
            while current_avg < min_accuracy and replace_idx < len(selected) and high_idx < len(available_high):
                old_sample = selected[replace_idx]
                new_sample = available_high[high_idx]
                
                # 只替换正确率<85%的样本
                if old_sample['accuracy'] < 0.85:
                    test_avg = (sum(r['accuracy'] for r in selected) - old_sample['accuracy'] + new_sample['accuracy']) / len(selected)
                    
                    if test_avg <= max_accuracy:  # 确保不会升太高
                        selected[replace_idx] = new_sample
                        current_avg = calculate_avg_acc(selected)
                        high_idx += 1
                
                replace_idx += 1
            
            log(f"  算法{algo_id}: 调整后正确率 {current_avg*100:.1f}%")
    
    # 第三步：处理负样本（如果正样本数量不足）
    current_count = len(selected)
    current_negative = sum(1 for r in selected if len(r['ground_truths']) == 0)
    
    if current_count < target_count:
        remaining_count = target_count - current_count
        allowed_negative = max_negative_count - current_negative
        
        if allowed_negative > 0 and negative_samples:
            # 优先选择真负样本（无检测结果，正确率=100%）
            neg_true_negative = [r for r in negative_samples if len(r['detections']) == 0]
            # 假阳性样本（有误检，正确率=0%）
            neg_false_positive = [r for r in negative_samples if len(r['detections']) > 0]
            
            neg_to_add = []
            
            # 根据当前正确率决定选择哪种负样本
            current_avg = calculate_avg_acc(selected) if selected else 0.5
            
            if current_avg > target_avg:
                # 正确率偏高，优先选择假阳性（会降低正确率）
                if neg_false_positive:
                    neg_to_add.extend(neg_false_positive[:min(len(neg_false_positive), allowed_negative)])
                if len(neg_to_add) < min(remaining_count, allowed_negative) and neg_true_negative:
                    remaining_neg = min(remaining_count, allowed_negative) - len(neg_to_add)
                    neg_to_add.extend(neg_true_negative[:remaining_neg])
            else:
                # 正确率适中或偏低，优先选择真负样本（正确率=100%）
                if neg_true_negative:
                    neg_to_add.extend(neg_true_negative[:min(len(neg_true_negative), allowed_negative)])
                if len(neg_to_add) < min(remaining_count, allowed_negative) and neg_false_positive:
                    remaining_neg = min(remaining_count, allowed_negative) - len(neg_to_add)
                    neg_to_add.extend(neg_false_positive[:remaining_neg])
            
            selected.extend(neg_to_add[:min(remaining_count, allowed_negative)])
    
    # 最终统计
    final_pos_count = sum(1 for r in selected if len(r['ground_truths']) > 0)
    final_neg_count = len(selected) - final_pos_count
    final_acc = sum(r['accuracy'] for r in selected) / len(selected) if selected else 0
    
    # 显示正确率分布
    acc_dist = {'>=98%': 0, '95-98%': 0, '90-95%': 0, '85-90%': 0, '<85%': 0}
    for r in selected:
        if r['accuracy'] >= 0.98:
            acc_dist['>=98%'] += 1
        elif r['accuracy'] >= 0.95:
            acc_dist['95-98%'] += 1
        elif r['accuracy'] >= 0.90:
            acc_dist['90-95%'] += 1
        elif r['accuracy'] >= 0.85:
            acc_dist['85-90%'] += 1
        else:
            acc_dist['<85%'] += 1
    
    log(f"  算法{algo_id}: 最终选择 正样本 {final_pos_count} 张, 负样本 {final_neg_count} 张")
    log(f"  算法{algo_id}: 正确率分布 - {acc_dist}")
    log(f"  算法{algo_id}: 最终平均正确率 {final_acc*100:.1f}% (目标: {min_accuracy*100:.0f}%-{max_accuracy*100:.0f}%)")
    
    # 创建输出子目录
    subdirs = ['images', 'detect_images', 'real_images', 'detect_labels', 'real_labels']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 保存文件
    saved_count = 0
    
    for result in selected:
        img_name = result['name']
        img_path = result['path']
        base_name = Path(img_name).stem
        
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            
            img_h, img_w = result['height'], result['width']
            
            orig_out_path = os.path.join(output_dir, 'images', img_name)
            cv2.imencode(Path(img_name).suffix, image)[1].tofile(orig_out_path)
            
            detect_image = draw_detections(image.copy(), result['detections'], algo_id)
            detect_out_path = os.path.join(output_dir, 'detect_images', img_name)
            cv2.imencode(Path(img_name).suffix, detect_image)[1].tofile(detect_out_path)
            
            real_image = draw_ground_truth(image.copy(), result['ground_truths'], algo_id)
            real_out_path = os.path.join(output_dir, 'real_images', img_name)
            cv2.imencode(Path(img_name).suffix, real_image)[1].tofile(real_out_path)
            
            detect_label = detections_to_yolo_format(result['detections'], img_w, img_h)
            detect_label_path = os.path.join(output_dir, 'detect_labels', f'{base_name}.txt')
            with open(detect_label_path, 'w', encoding='utf-8') as f:
                f.write(detect_label)
            
            real_label_src = os.path.join(label_dir, f'{base_name}.txt')
            real_label_dst = os.path.join(output_dir, 'real_labels', f'{base_name}.txt')
            if os.path.exists(real_label_src):
                shutil.copy2(real_label_src, real_label_dst)
            
            saved_count += 1
        except Exception as e:
            continue
    
    saved_accuracies = [r['accuracy'] for r in selected[:saved_count]]
    total_accuracy = sum(saved_accuracies) / len(saved_accuracies) if saved_accuracies else 0
    
    return saved_count, total_accuracy


# ============================================================
# 查找模型文件
# ============================================================
def find_model_path(algo_id: int, model_type: str) -> Optional[str]:
    """查找算法对应的模型文件"""
    if model_type == 'yolov11_720':
        possible_names = [
            f'{algo_id}_720.pt',
            f'{algo_id}_720_bs1.pt',
            f'{algo_id}.pt',
        ]
    else:
        possible_names = [
            f'{algo_id}.pt',
            f'{algo_id}_best.pt',
            f'{algo_id}_last.pt',
            f'algo_{algo_id}.pt',
            f'best_{algo_id}.pt',
        ]
    
    for name in possible_names:
        path = os.path.join(MODEL_DIR, name)
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
    print("  YOLO 批量自动化处理工具 (4×GPU并行) - V2")
    print("=" * 70)
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  配置文件: {YAML_CONFIG_PATH}")
    print(f"  GPU数量: {NUM_GPUS}")
    print(f"  GPU编号: {GPU_IDS}")
    print(f"  目标保存数量: {TARGET_COUNT}")
    print(f"  目标正确率范围: {MIN_ACCURACY * 100:.0f}% - {MAX_ACCURACY * 100:.0f}%")
    print(f"  负样本最大比例: {MAX_NEGATIVE_RATIO * 100}%")
    print("-" * 70)
    print("  支持模型: YOLOv5 (旧版) / YOLOv11 (ultralytics)")
    print("  根据 algo_classes.yaml 自动选择加载方式")
    print("-" * 70)
    print("  提示: YOLOv5模型首次加载会自动下载代码，请保持网络畅通")
    print("=" * 70)
    
    print(f"\n  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA设备数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU{i}: {torch.cuda.get_device_name(i)}")
    print("=" * 70)
    
    # 显示模型类型配置
    print("\n模型类型配置 (来自yaml):")
    for algo_id in sorted(MODEL_TYPES.keys()):
        if algo_id in ALGO_PATHS:
            print(f"  算法 {algo_id}: {MODEL_TYPES[algo_id]}")
    
    # 收集任务
    tasks = []
    skipped = []
    
    for algo_id in sorted(ALGO_PATHS.keys()):
        if algo_id == 13:
            skipped.append((algo_id, "变化检测算法"))
            continue
        
        paths = ALGO_PATHS[algo_id]
        model_type = MODEL_TYPES.get(algo_id, 'yolov5')
        model_path = find_model_path(algo_id, model_type)
        
        if not model_path:
            skipped.append((algo_id, f"模型文件不存在 (.pt)"))
            continue
        
        if not os.path.exists(paths['image_dir']):
            skipped.append((algo_id, f"图片目录不存在"))
            continue
        
        if not os.path.exists(paths['label_dir']):
            skipped.append((algo_id, f"标签目录不存在"))
            continue
        
        tasks.append({
            'algo_id': algo_id,
            'algo_name': paths.get('name', ALGO_CONFIG.get(algo_id, {}).get('name', f'算法{algo_id}')),
            'model_path': model_path,
            'model_type': model_type,
            'image_dir': paths['image_dir'],
            'label_dir': paths['label_dir'],
            'output_dir': paths['output_dir']
        })
    
    print(f"\n待处理算法: {len(tasks)} 个")
    for t in tasks:
        print(f"  - 算法 {t['algo_id']}: {t['algo_name']} ({t['model_type']})")
    
    if skipped:
        print(f"\n跳过算法: {len(skipped)} 个")
        for algo_id, reason in skipped:
            print(f"  - 算法 {algo_id}: {reason}")
    
    if not tasks:
        print("\n没有可处理的算法任务，程序退出。")
        return
    
    print("\n" + "=" * 70)
    print("  开始处理...")
    print("=" * 70 + "\n")
    
    t_total_start = time.time()
    
    all_results = []
    task_queue = list(tasks)
    
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=NUM_GPUS, mp_context=ctx) as executor:
        futures = {}
        gpu_available = list(GPU_IDS)
        
        while gpu_available and task_queue:
            gpu_id = gpu_available.pop(0)
            task = task_queue.pop(0)
            
            future = executor.submit(
                process_single_algorithm,
                gpu_id,
                task['algo_id'],
                task['model_path'],
                task['model_type'],
                task['image_dir'],
                task['label_dir'],
                task['output_dir'],
                DEFAULT_CONF,
                DEFAULT_IOU,
                ALGO_CONFIG,
                YAML_CONFIG_PATH
            )
            futures[future] = (gpu_id, task)
            log(f"提交任务: 算法{task['algo_id']} -> GPU{gpu_id}")
        
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
                        'success': False,
                        'error': str(e)
                    })
                
                if task_queue:
                    next_task = task_queue.pop(0)
                    next_future = executor.submit(
                        process_single_algorithm,
                        gpu_id,
                        next_task['algo_id'],
                        next_task['model_path'],
                        next_task['model_type'],
                        next_task['image_dir'],
                        next_task['label_dir'],
                        next_task['output_dir'],
                        DEFAULT_CONF,
                        DEFAULT_IOU,
                        ALGO_CONFIG,
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
    
    print(f"\n总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"成功: {success_count} 个, 失败: {fail_count} 个")
    
    print("\n详细结果:")
    print("-" * 90)
    print(f"{'算法ID':<8} {'算法名称':<18} {'模型类型':<12} {'图片数':<8} {'正确率':<10} {'保存数':<8} {'状态':<8}")
    print("-" * 90)
    
    for r in sorted(all_results, key=lambda x: x.get('algo_id', 0)):
        if r.get('success', False):
            model_type = MODEL_TYPES.get(r['algo_id'], 'yolov5')
            print(f"{r['algo_id']:<8} {r.get('algo_name', '-'):<18} {model_type:<12} "
                  f"{r.get('total_images', 0):<8} {r.get('avg_accuracy', 0)*100:>6.1f}%    "
                  f"{r.get('saved_count', 0):<8} {'✓ 成功':<8}")
        else:
            print(f"{r.get('algo_id', '?'):<8} {'-':<18} {'-':<12} {'-':<8} {'-':<10} {'-':<8} {'✗ 失败':<8}")
    
    print("-" * 90)
    
    # 保存报告
    report_path = os.path.join(MODEL_DIR, f'batch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO 批量自动化处理报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {total_elapsed:.1f} 秒\n")
            f.write(f"成功: {success_count} 个, 失败: {fail_count} 个\n\n")
            
            for r in sorted(all_results, key=lambda x: x.get('algo_id', 0)):
                if r.get('success', False):
                    f.write(f"算法 {r['algo_id']}: {r.get('algo_name', '-')}\n")
                    f.write(f"  - 模型类型: {MODEL_TYPES.get(r['algo_id'], 'yolov5')}\n")
                    f.write(f"  - 处理图片: {r.get('total_images', 0)}\n")
                    f.write(f"  - 平均正确率: {r.get('avg_accuracy', 0)*100:.1f}%\n")
                    f.write(f"  - 保存数量: {r.get('saved_count', 0)}\n")
                    f.write(f"  - 耗时: {r.get('elapsed_time', 0):.1f}秒\n\n")
                else:
                    f.write(f"算法 {r.get('algo_id', '?')}: 失败 - {r.get('error', '未知')}\n\n")
        
        print(f"\n报告已保存: {report_path}")
    except Exception as e:
        print(f"\n保存报告失败: {e}")
    
    print("\n所有任务已完成！")


if __name__ == '__main__':
    main()