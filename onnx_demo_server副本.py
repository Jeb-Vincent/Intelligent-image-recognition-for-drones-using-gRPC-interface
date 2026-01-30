#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 可视化演示服务 (Windows)

功能:
- 扫描指定目录下的模型和测试图片
- 提供 REST API 进行推理
- 支持 YOLOv5 和 YOLOv11 模型自动识别
- 前端可视化展示

使用方法:
    直接运行: python onnx_demo_server副本.py
    配置在下方 CONFIG 部分修改
"""

import os
import sys
import base64
import time
import json
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

# 服务配置
SERVER_HOST = '0.0.0.0'  # 监听地址，0.0.0.0 表示所有网卡
SERVER_PORT = 8082  # 服务端口

# 目录配置
MODEL_DIR = './'  # ONNX 模型目录（当前目录）
IMAGE_DIR = './test_images'  # 测试图片目录

# 检测参数默认值
DEFAULT_CONF = 0.25  # 默认置信度阈值
DEFAULT_IOU = 0.45  # 默认 IOU 阈值

# ============================================================

# 算法配置 (与 NPU 服务保持一致)
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

# 模型类型配置
MODEL_TYPES = {
    # yolov11_720 (输入 736)
    7: "yolov11_720", 8: "yolov11_720", 10: "yolov11_720", 12: "yolov11_720",
    14: "yolov11_720", 15: "yolov11_720", 16: "yolov11_720", 18: "yolov11_720",
    # yolov11 (输入 640)
    6: "yolov11",
    # 变化检测
    13: "change_detection",
    # 其他默认 yolov5
}


# ============================================================
# ONNX 变化检测器
# ============================================================
class ONNXChangeDetector:
    """ONNX 变化检测器 (SNUNet)"""

    def __init__(self, model_path: str, algo_id: int = 13):
        self.model_path = model_path
        self.algo_id = algo_id
        self.input_size = 256

        # 加载 ONNX 模型
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[ONNX] 加载变化检测模型: {model_path}")
        print(f"       输入: {self.input_names}")
        print(f"       输出: {self.output_names}")
        for inp in self.session.get_inputs():
            print(f"       {inp.name}: shape={inp.shape}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理单张图片"""
        # 调整大小
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        # 归一化到 [0, 1]
        img_norm = img_chw.astype(np.float32) / 255.0
        # 添加 batch 维度
        return np.expand_dims(img_norm, axis=0)

    def detect(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        执行变化检测

        Args:
            image1: 第一张图片 (BGR) - 时相1
            image2: 第二张图片 (BGR) - 时相2

        Returns:
            (mask_image, change_ratio, infer_time)
        """
        orig_h, orig_w = image1.shape[:2]

        # 预处理
        input1 = self.preprocess(image1)
        input2 = self.preprocess(image2)

        # 推理
        t0 = time.time()

        # SNUNet 有两个独立输入: img1, img2
        if len(self.input_names) == 2:
            inputs = {
                self.input_names[0]: input1,
                self.input_names[1]: input2
            }
        else:
            combined = np.concatenate([input1, input2], axis=1)
            inputs = {self.input_names[0]: combined}

        outputs = self.session.run(None, inputs)
        infer_time = time.time() - t0

        # 后处理
        # SNUNet 输出: [1, 2, H, W] (二分类 logits)
        output = outputs[0]
        print(f"[变化检测] 输出形状: {output.shape}, 范围: [{output.min():.3f}, {output.max():.3f}]")

        # 使用 argmax 获取预测类别
        if output.ndim == 4 and output.shape[1] == 2:
            # [1, 2, H, W] -> argmax -> [1, H, W]
            pred = np.argmax(output, axis=1)[0]  # [H, W]
            mask_binary = (pred == 1).astype(np.uint8) * 255
        elif output.ndim == 4 and output.shape[1] == 1:
            # [1, 1, H, W] -> sigmoid 后阈值化
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

        print(f"[变化检测] mask 形状: {mask_binary.shape}, 变化像素: {np.sum(mask_binary > 0)}")

        # 调整回原始大小
        mask_resized = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 计算变化比例
        change_ratio = np.sum(mask_resized > 0) / (orig_h * orig_w)

        # 创建彩色 mask
        mask_color = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        mask_color[mask_resized > 0] = [0, 0, 255]  # 红色表示变化区域 (BGR)

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

        # 根据模型类型确定输入尺寸
        if self.model_type == "yolov11_720":
            self.input_size = 736
        elif self.model_type == "change_detection":
            self.input_size = 256
        else:
            self.input_size = 640

        # 加载 ONNX 模型
        providers = ['CPUExecutionProvider']
        # 如果有 CUDA，优先使用
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输出形状来判断模型类型
        output_shape = self.session.get_outputs()[0].shape
        self._detect_model_format(output_shape)

        print(f"[ONNX] 加载模型: algo={algo_id}, path={model_path}")
        print(f"       类型={self.model_type}, 输入={self.input_size}, 格式={self._format}")

    def _detect_model_format(self, output_shape):
        """检测模型输出格式"""
        # output_shape 可能是 [1, 25200, 85] (YOLOv5) 或 [1, 84, 8400] (YOLOv8/v11)
        if len(output_shape) == 3:
            dim1 = output_shape[1]
            dim2 = output_shape[2]

            if isinstance(dim1, int) and isinstance(dim2, int):
                if dim1 > dim2:
                    # YOLOv5: [1, num_preds, num_features]
                    self._format = "yolov5"
                    self._has_objectness = True
                    self._transpose = False
                else:
                    # YOLOv8/v11: [1, num_features, num_preds]
                    self._format = "yolov11"
                    self._has_objectness = False
                    self._transpose = True
            else:
                # 动态形状，根据配置判断
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
        """图像预处理 (letterbox)"""
        orig_h, orig_w = image.shape[:2]
        input_size = self.input_size

        # 计算缩放比例
        r = min(input_size / orig_h, input_size / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)

        # 缩放图像
        img_resized = cv2.resize(image, (new_w, new_h))

        # 计算 padding
        dw, dh = (input_size - new_w) / 2, (input_size - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 添加边框
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换颜色空间和格式
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        img_norm = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        return img_batch, (orig_h, orig_w, r, top, left)

    def detect(self, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[Dict]:
        """执行目标检测"""
        # 预处理
        input_data, img_info = self.preprocess(image)

        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0]

        # 后处理
        return self._postprocess(output, img_info, conf_thres, iou_thres)

    def _postprocess(self, output: np.ndarray, img_info: tuple,
                     conf_thres: float, iou_thres: float) -> List[Dict]:
        """后处理"""
        orig_h, orig_w, ratio, pad_top, pad_left = img_info

        # 去除 batch 维度
        if output.ndim == 3:
            output = output[0]

        # 根据格式处理
        if self._transpose:
            # YOLOv11: [num_features, num_preds] -> [num_preds, num_features]
            predictions = output.T
        else:
            # YOLOv5: [num_preds, num_features]
            predictions = output

        # 提取边界框和置信度
        boxes = predictions[:, :4].copy()

        if self._has_objectness:
            # YOLOv5: 有 objectness
            obj_conf = predictions[:, 4]
            class_scores = predictions[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            class_conf = np.max(class_scores, axis=1)
            scores = obj_conf * class_conf
        else:
            # YOLOv11: 无 objectness
            class_scores = predictions[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = np.max(class_scores, axis=1)

        # 置信度过滤
        mask = scores > conf_thres
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

        # 去除 padding
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top

        # 缩放回原图
        boxes_xyxy /= ratio

        # 过滤无效框
        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes_xyxy) == 0:
            return []

        # NMS
        keep = self._nms(boxes_xyxy, scores, iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # 裁剪到图像边界
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # 构建结果
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
# 模型管理器
# ============================================================
class ModelManager:
    """模型管理器"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models: Dict[int, Any] = {}  # 可以是 ONNXYOLODetector 或 ONNXChangeDetector
        self._scan_models()

    def _scan_models(self):
        """扫描模型文件"""
        if not os.path.exists(self.model_dir):
            print(f"[警告] 模型目录不存在: {self.model_dir}")
            return

        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.onnx'):
                continue

            filepath = os.path.join(self.model_dir, filename)

            # 解析算法 ID
            # 支持格式: {algo_id}.onnx, {algo_id}_720.onnx, {algo_id}_bs1.onnx
            base_name = filename.replace('.onnx', '')
            base_name = base_name.replace('_720', '').replace('_bs1', '')

            try:
                algo_id = int(base_name)
            except ValueError:
                continue

            # 720 模型优先
            if '_720' in filename and algo_id in self.models:
                # 替换为 720 模型
                pass
            elif algo_id in self.models:
                continue

            try:
                # 根据模型类型选择检测器
                model_type = MODEL_TYPES.get(algo_id, "yolov5")
                if model_type == "change_detection":
                    self.models[algo_id] = ONNXChangeDetector(filepath, algo_id)
                else:
                    self.models[algo_id] = ONNXYOLODetector(filepath, algo_id)
            except Exception as e:
                print(f"[错误] 加载模型失败: {filename}, {e}")

        print(f"[模型管理器] 加载了 {len(self.models)} 个模型")

    def get_detector(self, algo_id: int):
        """获取检测器"""
        return self.models.get(algo_id)

    def is_change_detection(self, algo_id: int) -> bool:
        """判断是否为变化检测模型"""
        return MODEL_TYPES.get(algo_id) == "change_detection"

    def get_available_algos(self) -> List[Dict]:
        """获取可用算法列表"""
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
    """图片文件夹管理器"""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, image_dir: str):
        self.image_dir = image_dir

    def get_folders(self) -> List[Dict]:
        """获取所有子文件夹，如果当前目录有图片也显示"""
        if not os.path.exists(self.image_dir):
            return []

        folders = []

        # 检查当前目录是否有图片
        root_image_count = sum(1 for f in os.listdir(self.image_dir)
                               if os.path.isfile(os.path.join(self.image_dir, f)) and
                               Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS)
        if root_image_count > 0:
            folders.append({
                "name": "📷 当前目录",
                "path": "__ROOT__",  # 使用特殊标识
                "image_count": root_image_count
            })

        # 获取子文件夹
        for name in sorted(os.listdir(self.image_dir)):
            folder_path = os.path.join(self.image_dir, name)
            if os.path.isdir(folder_path):
                # 统计图片数量
                image_count = sum(1 for f in os.listdir(folder_path)
                                  if Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS)
                if image_count > 0:  # 只显示有图片的文件夹
                    folders.append({
                        "name": name,
                        "path": name,
                        "image_count": image_count
                    })

        return folders

    def get_images(self, folder_name: str) -> List[Dict]:
        """获取文件夹中的图片列表"""
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
                # 获取文件大小
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

    def get_image_path(self, relative_path: str) -> Optional[str]:
        """获取图片完整路径"""
        full_path = os.path.join(self.image_dir, relative_path)
        if os.path.exists(full_path):
            return full_path
        return None

    @staticmethod
    def _format_size(size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


# ============================================================
# Flask 应用
# ============================================================
app = Flask(__name__, static_folder=None)
CORS(app)

model_manager: ModelManager = None
image_manager: ImageFolderManager = None


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('.', 'onnx_demo副本.html')


@app.route('/api/config', methods=['POST', 'GET'])
def api_config():
    """获取或更新配置"""
    global model_manager, image_manager

    if request.method == 'GET':
        return jsonify({
            "code": 200,
            "data": {
                "model_dir": model_manager.model_dir if model_manager else MODEL_DIR,
                "image_dir": image_manager.image_dir if image_manager else IMAGE_DIR
            }
        })

    # POST - 更新配置
    try:
        data = request.get_json()
        new_model_dir = data.get('model_dir')
        new_image_dir = data.get('image_dir')

        # 更新模型管理器
        if new_model_dir:
            if os.path.exists(new_model_dir):
                model_manager = ModelManager(new_model_dir)
                print(f"[配置] 模型目录已更新: {new_model_dir}")
            else:
                return jsonify({"code": 400, "message": f"模型目录不存在: {new_model_dir}"})

        # 更新图片管理器
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
    """获取可用算法列表"""
    return jsonify({
        "code": 200,
        "data": model_manager.get_available_algos()
    })


@app.route('/api/folders')
def get_folders():
    """获取图片文件夹列表"""
    return jsonify({
        "code": 200,
        "data": image_manager.get_folders()
    })


@app.route('/api/images/<path:folder_name>')
def get_images(folder_name: str):
    """获取文件夹中的图片列表"""
    return jsonify({
        "code": 200,
        "data": image_manager.get_images(folder_name)
    })


@app.route('/api/image/<path:image_path>')
def get_image(image_path: str):
    """获取图片文件"""
    full_path = image_manager.get_image_path(image_path)
    if full_path:
        return send_file(full_path)
    return jsonify({"code": 404, "message": "图片不存在"}), 404


@app.route('/api/thumb/<path:image_path>')
def get_thumbnail(image_path: str):
    """获取图片缩略图"""
    full_path = image_manager.get_image_path(image_path)
    if not full_path:
        return jsonify({"code": 404, "message": "图片不存在"}), 404

    try:
        # 读取图片（支持中文路径）
        image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"code": 400, "message": "无法读取图片"}), 400

        # 生成缩略图 (最大 100x100)
        h, w = image.shape[:2]
        max_size = 100
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

        thumb = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 编码为 JPEG
        _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])

        return send_file(
            BytesIO(buffer.tobytes()),
            mimetype='image/jpeg'
        )
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect():
    """执行目标检测"""
    data = request.json

    algo_id = data.get('algorithm_id')
    image_path = data.get('image_path')
    conf_threshold = data.get('conf_threshold', DEFAULT_CONF)
    iou_threshold = data.get('iou_threshold', DEFAULT_IOU)

    if not algo_id:
        return jsonify({"code": 400, "message": "缺少 algorithm_id"}), 400

    if not image_path:
        return jsonify({"code": 400, "message": "缺少 image_path"}), 400

    # 获取检测器
    detector = model_manager.get_detector(algo_id)
    if not detector:
        return jsonify({"code": 404, "message": f"算法 {algo_id} 不存在"}), 404

    # 获取图片
    full_path = image_manager.get_image_path(image_path)
    if not full_path:
        return jsonify({"code": 404, "message": "图片不存在"}), 404

    # 读取图片（支持中文路径）
    try:
        # Windows 中文路径需要特殊处理
        image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400

    if image is None:
        return jsonify({"code": 400, "message": "无法读取图片，请检查图片格式"}), 400

    img_h, img_w = image.shape[:2]

    # 执行检测
    t0 = time.time()
    try:
        detections = detector.detect(image, conf_threshold, iou_threshold)
    except Exception as e:
        return jsonify({"code": 500, "message": f"检测失败: {e}"}), 500

    detect_time = time.time() - t0

    # 绘制检测结果
    result_image = draw_detections(image, detections)

    # 编码为 base64
    _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result_b64 = base64.b64encode(buffer).decode('utf-8')

    # 构建响应
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
        return jsonify({"code": 400, "message": "需要提供两张图片路径 (image_path1, image_path2)"}), 400

    # 获取检测器
    detector = model_manager.get_detector(algo_id)
    if not detector:
        return jsonify({"code": 404, "message": f"算法 {algo_id} 不存在"}), 404

    if not model_manager.is_change_detection(algo_id):
        return jsonify({"code": 400, "message": f"算法 {algo_id} 不是变化检测模型"}), 400

    # 获取图片1
    full_path1 = image_manager.get_image_path(image_path1)
    if not full_path1:
        return jsonify({"code": 404, "message": f"图片1不存在: {image_path1}"}), 404

    # 获取图片2
    full_path2 = image_manager.get_image_path(image_path2)
    if not full_path2:
        return jsonify({"code": 404, "message": f"图片2不存在: {image_path2}"}), 404

    # 读取图片
    try:
        image1_orig = cv2.imdecode(np.fromfile(full_path1, dtype=np.uint8), cv2.IMREAD_COLOR)
        image2_orig = cv2.imdecode(np.fromfile(full_path2, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400

    if image1_orig is None or image2_orig is None:
        return jsonify({"code": 400, "message": "无法读取图片"}), 400

    # 变化检测需要统一尺寸到 256x256
    input_size = 256
    image1 = cv2.resize(image1_orig, (input_size, input_size))
    image2 = cv2.resize(image2_orig, (input_size, input_size))

    # 执行变化检测
    try:
        mask_color, change_ratio, detect_time = detector.detect(image1, image2)
    except Exception as e:
        return jsonify({"code": 500, "message": f"检测失败: {e}"}), 500

    # 创建叠加结果图
    overlay = image2.copy()
    overlay[mask_color[:, :, 2] > 0] = [0, 0, 255]  # 红色标记变化区域
    blended = cv2.addWeighted(image2, 0.7, overlay, 0.3, 0)

    # 编码结果图
    _, buffer1 = cv2.imencode('.jpg', image1, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, buffer2 = cv2.imencode('.jpg', image2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, buffer_mask = cv2.imencode('.jpg', mask_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, buffer_blend = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 90])

    image1_b64 = base64.b64encode(buffer1).decode('utf-8')
    image2_b64 = base64.b64encode(buffer2).decode('utf-8')
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
            "change_ratio": round(change_ratio * 100, 2),  # 百分比
            "detect_time": round(detect_time, 3),
            "image1": f"data:image/jpeg;base64,{image1_b64}",
            "image2": f"data:image/jpeg;base64,{image2_b64}",
            "mask_image": f"data:image/jpeg;base64,{mask_b64}",
            "result_image": f"data:image/jpeg;base64,{blend_b64}"
        }
    })


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """在图片上绘制检测结果（支持中文）"""
    from PIL import Image, ImageDraw, ImageFont

    # 转换为 PIL 图像
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_image)

    # 尝试加载中文字体
    font = None
    font_size = 20
    # Windows 常见中文字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
        "/System/Library/Fonts/PingFang.ttc",  # macOS
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

    if font is None:
        # 使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # 颜色列表 (RGB)
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    ]

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = det["class_id"]
        class_name = det["class_name"]
        confidence = det["confidence"]

        # 选择颜色
        color = colors[class_id % len(colors)]

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签
        label = f"{class_name} {confidence:.2f}"

        # 获取文本大小
        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except:
            text_w, text_h = draw.textsize(label, font=font) if hasattr(draw, 'textsize') else (len(label) * 10, 20)

        # 绘制标签背景
        label_y = max(y1 - text_h - 8, 0)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 6], fill=color)

        # 绘制标签文字
        draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255), font=font)

    # 转换回 OpenCV 格式
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def main():
    global model_manager, image_manager

    print("=" * 60)
    print("ONNX 可视化演示服务")
    print("=" * 60)
    print(f"模型目录: {MODEL_DIR}")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"服务地址: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"默认置信度: {DEFAULT_CONF}")
    print(f"默认IOU: {DEFAULT_IOU}")
    print("=" * 60)

    # 检查目录
    if not os.path.exists(MODEL_DIR):
        print(f"\n[警告] 模型目录不存在，正在创建: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[提示] 请将 .onnx 模型文件放入 {MODEL_DIR} 目录")

    if not os.path.exists(IMAGE_DIR):
        print(f"\n[警告] 图片目录不存在，正在创建: {IMAGE_DIR}")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print(f"[提示] 请在 {IMAGE_DIR} 目录下创建子文件夹并放入测试图片")

    # 初始化管理器
    model_manager = ModelManager(MODEL_DIR)
    image_manager = ImageFolderManager(IMAGE_DIR)

    # 启动服务
    print(f"\n🚀 服务已启动: http://localhost:{SERVER_PORT}")
    print(f"   浏览器访问上述地址即可使用\n")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()