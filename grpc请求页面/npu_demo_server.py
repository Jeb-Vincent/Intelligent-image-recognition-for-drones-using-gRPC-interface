#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU gRPC 代理服务 (Windows/Linux)

功能:
- 提供 HTTP REST API 接口
- 转发请求到华为 NPU gRPC 服务
- 支持图片文件夹浏览
- 支持中文路径和中文标签

使用方法:
    直接运行: python npu_demo_server.py
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
import grpc
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ============================================================
# ★★★ 配置区域 - 在此处修改配置 ★★★
# ============================================================

# 本地 HTTP 服务配置
SERVER_HOST = '0.0.0.0'  # 监听地址
SERVER_PORT = 8080  # 本地服务端口

# 远程 NPU gRPC 服务配置
GRPC_HOST = '172.18.8.11'  # NPU 服务器地址
GRPC_PORT = 8000  # NPU gRPC 端口
API_KEY = 'api-key'  # API 密钥

# 图片目录配置
IMAGE_DIR = './test_images'  # 测试图片目录

# 检测参数默认值
DEFAULT_CONF = 0.25  # 默认置信度阈值
DEFAULT_IOU = 0.45  # 默认 IOU 阈值

# gRPC 超时时间（秒）
GRPC_TIMEOUT = 30


# ============================================================

# 动态生成 gRPC 代码（如果 proto 文件存在）
def generate_grpc_code():
    """尝试从 proto 文件生成 gRPC 代码"""
    proto_file = 'detection.proto'
    if os.path.exists(proto_file):
        try:
            import grpc_tools.protoc
            grpc_tools.protoc.main([
                'grpc_tools.protoc',
                f'--proto_path=.',
                f'--python_out=.',
                f'--grpc_python_out=.',
                proto_file
            ])
            print(f"[gRPC] 已从 {proto_file} 生成代码")
        except Exception as e:
            print(f"[gRPC] 生成代码失败: {e}")


# 尝试导入 gRPC 生成的代码
try:
    import detection_pb2
    import detection_pb2_grpc

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("[警告] 未找到 detection_pb2 模块，将使用模拟模式")

# ============================================================
# 算法配置（与 NPU 服务保持一致）
# ============================================================
ALGO_CONFIG = {
    1: {"name": "松线虫害识别", "classes": ["死亡", "重度患病", "轻度患病"]},
    2: {"name": "水污染检测", "classes": ["水污染", "漂浮碎片", "废弃船只", "渔业和水产养殖", "垃圾"]},
    3: {"name": "水面垃圾检测", "classes": ["瓶子", "草", "树枝", "牛奶盒", "塑料袋", "塑料垃圾袋", "球", "叶子"]},
    4: {"name": "水域安全检测", "classes": ["忽略", "游泳者", "船", "水上摩托艇", "救生设备", "浮标"]},
    5: {"name": "车牌识别", "classes": ["车牌"]},
    6: {"name": "车辆识别", "classes": ["车辆"]},
    7: {"name": "路面病害识别",
        "classes": ["龟裂", "纵向裂缝", "纵向修补块", "检查井井盖", "坑洞", "横向裂缝", "横向修补块"]},
    8: {"name": "城市部件检测",
        "classes": ["违规广告牌", "破损标识牌", "人行道杂物堆积", "施工路段", "褪色标识牌", "垃圾堆积", "涂鸦乱画",
                    "路面坑洞", "路面积沙", "建筑外立面破损"]},
    9: {"name": "人车检测", "classes": ["车", "人"]},
    10: {"name": "防溺水检测", "classes": ["水边钓鱼", "游泳溺水", "钓鱼伞", "船"]},
    11: {"name": "工程机械检测", "classes": ["起重机", "挖掘机", "拖拉机", "卡车"]},
    12: {"name": "秸秆焚烧检测", "classes": ["秸秆堆"]},
    13: {"name": "变化检测", "classes": ["无变化", "变化区域"]},
    14: {"name": "占道经营识别", "classes": ["占道经营"]},
    15: {"name": "城市违规检测",
         "classes": ["长椅", "商业垃圾", "非法倾倒点", "绿地", "孔洞", "泽西护栏", "地块", "原材料", "生活垃圾"]},
    16: {"name": "裸土垃圾检测", "classes": ["垃圾", "裸土"]},
    17: {"name": "违建识别", "classes": ["蓝色天篷", "其他违建", "改装绿色小屋"]},
    18: {"name": "烟火检测", "classes": ["烟雾", "火"]},
    19: {"name": "光伏板检测", "classes": ["有缺陷的光伏电池"]},
    20: {"name": "行人车辆检测", "classes": ["人", "车", "自行车"]},
    21: {"name": "墙体病害检测", "classes": ["墙体腐蚀", "墙体开裂", "墙体劣化", "墙模", "墙面污渍"]},
    22: {"name": "罂粟识别", "classes": ["罂粟"]},
    23: {"name": "农作物识别", "classes": ["作物倒伏"]},
    24: {"name": "蓝藻检测", "classes": ["蓝藻"]},
    25: {"name": "船只检测", "classes": ["船只"]},
}


# ============================================================
# gRPC 客户端
# ============================================================
class GRPCClient:
    """gRPC 客户端封装"""

    def __init__(self, host: str, port: int, api_key: str):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.channel = None
        self.stub = None

    def connect(self):
        """建立连接"""
        if not GRPC_AVAILABLE:
            return False

        try:
            target = f"{self.host}:{self.port}"
            self.channel = grpc.insecure_channel(
                target,
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ]
            )
            self.stub = detection_pb2_grpc.DetectionServiceStub(self.channel)
            return True
        except Exception as e:
            print(f"[gRPC] 连接失败: {e}")
            return False

    def _get_metadata(self):
        """获取请求元数据"""
        return [('x-api-key', self.api_key)]

    def health_check(self) -> Dict:
        """健康检查"""
        if not self.stub:
            self.connect()

        try:
            request = detection_pb2.HealthRequest()
            response = self.stub.HealthCheck(
                request,
                metadata=self._get_metadata(),
                timeout=5
            )
            return {
                "status": response.status,
                "device": response.device,
                "models_cached": response.models_cached
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_version(self) -> Dict:
        """获取版本信息"""
        if not self.stub:
            self.connect()

        try:
            request = detection_pb2.VersionRequest()
            response = self.stub.GetVersion(
                request,
                metadata=self._get_metadata(),
                timeout=5
            )
            return {
                "version": response.version,
                "device": response.device,
                "algo_supported": list(response.algo_supported)
            }
        except Exception as e:
            return {"version": "unknown", "error": str(e)}

    def detect(self, image: np.ndarray, algorithm_id: int,
               conf_threshold: float = 0.25) -> Tuple[List[Dict], float]:
        """
        执行目标检测

        Returns:
            (detections, detect_time)
        """
        if not self.stub:
            self.connect()

        # 编码图片为 base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        # 构建请求
        request = detection_pb2.DetectRequest(
            algorithm_id=algorithm_id,
            image=image_b64,
            conf_threshold=conf_threshold
        )

        # 发送请求
        response = self.stub.Detect(
            request,
            metadata=self._get_metadata(),
            timeout=GRPC_TIMEOUT
        )

        if response.code != 200:
            raise ValueError(response.message)

        # 解析结果
        detections = []
        for det in response.data.detections:
            detections.append({
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": list(det.bbox)
            })

        return detections, response.data.detect_time

    def detect_change(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        """
        执行变化检测

        Args:
            image1: 第一张图片 (BGR)
            image2: 第二张图片 (BGR)

        Returns:
            (mask_image, change_ratio, detect_time, width, height)
        """
        if not self.stub:
            self.connect()

        # 编码图片为 base64 PNG
        _, buffer1 = cv2.imencode('.png', image1)
        _, buffer2 = cv2.imencode('.png', image2)
        image1_b64 = base64.b64encode(buffer1).decode('utf-8')
        image2_b64 = base64.b64encode(buffer2).decode('utf-8')

        # 构建请求
        request = detection_pb2.ChangeDetectRequest(
            image1=image1_b64,
            image2=image2_b64
        )

        # 发送请求
        response = self.stub.DetectChange(
            request,
            metadata=self._get_metadata(),
            timeout=GRPC_TIMEOUT
        )

        if response.code != 200:
            raise ValueError(response.message)

        # 解码 mask 图片
        mask_data = base64.b64decode(response.mask)
        mask_array = np.frombuffer(mask_data, dtype=np.uint8)
        mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

        return mask_image, response.change_ratio, response.detect_time, response.width, response.height


# ============================================================
# 图片文件夹管理器
# ============================================================
class ImageFolderManager:
    """图片文件夹管理器"""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def __init__(self, image_dir: str):
        self.image_dir = image_dir

    def get_folders(self) -> List[Dict]:
        """获取所有子文件夹"""
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
                "path": "__ROOT__",
                "image_count": root_image_count
            })

        # 获取子文件夹
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

grpc_client: GRPCClient = None
image_manager: ImageFolderManager = None


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('.', 'npu_demo.html')


@app.route('/api/config', methods=['POST', 'GET'])
def api_config():
    """获取或更新配置"""
    global image_manager, grpc_client

    if request.method == 'GET':
        return jsonify({
            "code": 200,
            "data": {
                "grpc_host": grpc_client.host if grpc_client else GRPC_HOST,
                "grpc_port": grpc_client.port if grpc_client else GRPC_PORT,
                "image_dir": image_manager.image_dir if image_manager else IMAGE_DIR
            }
        })

    # POST - 更新配置
    try:
        data = request.get_json()

        # 更新 gRPC 客户端
        new_grpc_host = data.get('grpc_host')
        new_grpc_port = data.get('grpc_port')
        if new_grpc_host or new_grpc_port:
            host = new_grpc_host or grpc_client.host
            port = int(new_grpc_port) if new_grpc_port else grpc_client.port
            grpc_client = GRPCClient(host, port, API_KEY)
            grpc_client.connect()
            print(f"[配置] gRPC 服务已更新: {host}:{port}")

        # 更新图片管理器
        new_image_dir = data.get('image_dir')
        if new_image_dir:
            if os.path.exists(new_image_dir):
                image_manager = ImageFolderManager(new_image_dir)
                print(f"[配置] 图片目录已更新: {new_image_dir}")
            else:
                return jsonify({"code": 400, "message": f"图片目录不存在: {new_image_dir}"})

        return jsonify({
            "code": 200,
            "message": "配置已更新"
        })
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)})


@app.route('/api/health')
def api_health():
    """健康检查"""
    result = grpc_client.health_check()
    return jsonify({"code": 200, "data": result})


@app.route('/api/algorithms')
def get_algorithms():
    """获取支持的算法列表"""
    algorithms = []
    for algo_id, config in sorted(ALGO_CONFIG.items()):
        algorithms.append({
            "id": algo_id,
            "name": config["name"],
            "classes": config["classes"],
            "type": "change_detection" if algo_id == 13 else "detection"
        })
    return jsonify({"code": 200, "data": algorithms})


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

        # 生成缩略图
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
    """执行目标检测"""
    data = request.json

    algo_id = data.get('algorithm_id')
    image_path = data.get('image_path')
    conf_threshold = data.get('conf_threshold', DEFAULT_CONF)

    if not algo_id:
        return jsonify({"code": 400, "message": "缺少 algorithm_id"}), 400

    if not image_path:
        return jsonify({"code": 400, "message": "缺少 image_path"}), 400

    # 获取图片
    full_path = image_manager.get_image_path(image_path)
    if not full_path:
        return jsonify({"code": 404, "message": "图片不存在"}), 404

    # 读取图片（支持中文路径）
    try:
        image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"code": 400, "message": f"读取图片失败: {e}"}), 400

    if image is None:
        return jsonify({"code": 400, "message": "无法读取图片"}), 400

    # 调用 gRPC 服务检测
    try:
        detections, detect_time = grpc_client.detect(image, algo_id, conf_threshold)
    except Exception as e:
        return jsonify({"code": 500, "message": f"检测失败: {e}"}), 500

    # 绘制检测结果
    result_img = draw_detections(image, detections)

    # 编码为 base64
    _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result_b64 = base64.b64encode(buffer).decode('utf-8')

    # 获取算法名称
    algo_name = ALGO_CONFIG.get(algo_id, {}).get("name", f"算法{algo_id}")

    return jsonify({
        "code": 200,
        "data": {
            "algorithm_id": algo_id,
            "algorithm_name": algo_name,
            "detections": detections,
            "total_count": len(detections),
            "detect_time": detect_time,
            "result_image": f"data:image/jpeg;base64,{result_b64}"
        }
    })


@app.route('/api/detect_change', methods=['POST'])
def detect_change():
    """执行变化检测（需要两张图片）"""
    data = request.json

    image_path1 = data.get('image_path1')
    image_path2 = data.get('image_path2')

    if not image_path1 or not image_path2:
        return jsonify({"code": 400, "message": "需要提供两张图片路径 (image_path1, image_path2)"}), 400

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

    # 调用 gRPC 变化检测服务
    try:
        mask_gray, change_ratio, detect_time, width, height = grpc_client.detect_change(image1, image2)
    except Exception as e:
        return jsonify({"code": 500, "message": f"检测失败: {e}"}), 500

    # 创建彩色 mask (红色表示变化)
    mask_color = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    if mask_gray is not None:
        mask_color[mask_gray > 127] = [0, 0, 255]  # BGR: 红色

    # 创建叠加结果图
    overlay = image2.copy()
    overlay[mask_gray > 127] = [0, 0, 255]
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

    return jsonify({
        "code": 200,
        "data": {
            "algorithm_id": 13,
            "algorithm_name": "变化检测",
            "image_width": input_size,
            "image_height": input_size,
            "change_ratio": round(change_ratio * 100, 2),
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

        color = colors[class_id % len(colors)]

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签
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

    # 转换回 OpenCV 格式
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def main():
    global grpc_client, image_manager

    print("=" * 60)
    print("NPU gRPC 代理服务")
    print("=" * 60)
    print(f"本地服务: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"远程 NPU: {GRPC_HOST}:{GRPC_PORT}")
    print(f"图片目录: {IMAGE_DIR}")
    print("=" * 60)

    # 检查目录
    if not os.path.exists(IMAGE_DIR):
        print(f"\n[警告] 图片目录不存在，正在创建: {IMAGE_DIR}")
        os.makedirs(IMAGE_DIR, exist_ok=True)

    # 初始化
    grpc_client = GRPCClient(GRPC_HOST, GRPC_PORT, API_KEY)
    image_manager = ImageFolderManager(IMAGE_DIR)

    # 尝试连接 gRPC 服务
    if GRPC_AVAILABLE:
        if grpc_client.connect():
            health = grpc_client.health_check()
            if health.get("status") == "ok":
                print(f"\n✅ 已连接到 NPU 服务: {health.get('device')}")
            else:
                print(f"\n⚠️ NPU 服务连接异常: {health}")
        else:
            print("\n⚠️ 无法连接到 NPU 服务，请检查网络")
    else:
        print("\n⚠️ gRPC 模块未加载，请确保 detection_pb2.py 存在")

    # 启动服务
    print(f"\n🚀 服务已启动: http://localhost:{SERVER_PORT}")
    print(f"   浏览器访问上述地址即可使用\n")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()