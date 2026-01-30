#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU Detection gRPC Service - Preload All Models
华为昇腾 NPU 推理服务（预加载所有模型到所有设备）

兼容 YOLOv5 和 YOLOv11 模型（自动识别）
支持 YOLOv11_720 模型（输入尺寸 736x736）

新增：算法13 - 变化检测模型（base64编码PNG输入输出）
"""

import base64
import time
import os
import sys
import resource
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent import futures
from collections import OrderedDict, defaultdict

import yaml
import cv2
import numpy as np
import acl
import grpc
from grpc_reflection.v1alpha import reflection


# ============================================================
# 系统资源限制调整
# ============================================================
def adjust_file_descriptor_limit(target_limit: int = 65535):
    """
    调整文件描述符限制，避免 'File descriptor limit reached' 错误
    
    Args:
        target_limit: 目标限制值，默认 65535
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[系统] 当前文件描述符限制: soft={soft}, hard={hard}")
        
        # 尝试调整到目标值，但不能超过 hard limit
        new_soft = min(target_limit, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"[系统] 文件描述符限制已调整: soft={new_soft}, hard={hard}")
        else:
            print(f"[系统] 文件描述符限制无需调整")
    except (ValueError, resource.error) as e:
        print(f"[警告] 无法调整文件描述符限制: {e}")
        print(f"[提示] 请手动执行: ulimit -n 65535 或修改 /etc/security/limits.conf")


# 车牌识别（静默导入）
try:
    from universal_plate_recognizer import UniversalPlateRecognizer
    _PLATE_RECOGNIZER_AVAILABLE = True
except ImportError:
    _PLATE_RECOGNIZER_AVAILABLE = False

import detection_pb2
import detection_pb2_grpc

# ============================================================
# 配置参数
# ============================================================
API_KEY = os.getenv("API_KEY", "api-key")
MODEL_DIR = os.getenv("MODEL_DIR", "/home/yolov11-4.0/modelv1")
DEFAULT_CONF = 0.25
MAX_DECODE_SIDE = int(os.getenv("MAX_DECODE_SIDE", "4096"))

# 算法类别配置文件路径
ALGO_CLASSES_CONFIG = os.getenv("ALGO_CLASSES_CONFIG", os.path.join(os.path.dirname(__file__), "algo_classes.yaml"))

# NPU 设备配置 - 支持环境变量控制
# NPU_DEVICE_COUNT: 可用的 NPU 设备总数，默认 16
# NPU_DEVICE_START: 起始设备 ID，默认 0
# MODEL_REPLICAS: 每个模型加载的副本数（分布在不同设备上），默认 8
_npu_device_count = int(os.getenv("NPU_DEVICE_COUNT", "16"))
_npu_device_start = int(os.getenv("NPU_DEVICE_START", "0"))
NPU_DEVICE_IDS = list(range(_npu_device_start, _npu_device_start + _npu_device_count))

# 每个模型的副本数（加载到多少个设备上）
MODEL_REPLICAS = int(os.getenv("MODEL_REPLICAS", "8"))

# 是否预加载所有模型到所有设备
PRELOAD_ALL_MODELS = os.getenv("PRELOAD_ALL_MODELS", "true").lower() == "true"

# gRPC 工作线程数，默认 16
GRPC_WORKERS = int(os.getenv("GRPC_WORKERS", "16"))

# 模型输入尺寸配置
MODEL_INPUT_SIZES = {
    'yolov5': 640,
    'yolov11': 640,
    'yolov11_720': 736,  # 720 调整为 32 的倍数
    'change_detection': 256,
}


# ============================================================
# 算法类别配置加载器
# ============================================================
class AlgoClassesConfig:
    """
    算法类别配置管理器
    
    从 YAML 文件加载算法配置，按算法ID隔离存储，
    避免不同算法间相同类别名的冲突。
    
    支持的模型类型 (model_type):
      - yolov5: YOLOv5 模型，输出形状 [25200, 5+num_classes]，有 objectness，输入 640
      - yolov11: YOLOv8/v11 模型，输出形状 [4+num_classes, 8400]，无 objectness，输入 640
      - yolov11_720: YOLOv11 模型，输入 736（720 调整为 32 倍数）
      - change_detection: 变化检测模型，特殊处理，输入 256
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        # 按算法ID存储：{algo_id: {class_id: {"name": str, "name_cn": str}}}
        self._algo_classes: Dict[int, Dict[int, Dict[str, str]]] = {}
        # 算法名称映射：{algo_id: algo_name}
        self._algo_names: Dict[int, str] = {}
        # 算法模型类型映射：{algo_id: model_type}
        self._algo_model_types: Dict[int, str] = {}
        self._load_config()
    
    def _load_config(self):
        """从 YAML 文件加载配置"""
        if not os.path.exists(self.config_path):
            print(f"[警告] 算法类别配置文件不存在: {self.config_path}")
            print("[警告] 将使用内置默认配置")
            self._use_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            algorithms = config.get('algorithms', {})
            
            for algo_id, algo_config in algorithms.items():
                algo_id = int(algo_id)
                self._algo_names[algo_id] = algo_config.get('name', f'算法{algo_id}')
                
                # 读取模型类型，默认为 yolov5
                self._algo_model_types[algo_id] = algo_config.get('model_type', 'yolov5')
                
                # 按算法ID独立存储类别，避免跨算法冲突
                self._algo_classes[algo_id] = {}
                classes = algo_config.get('classes', {})
                
                for class_id, class_info in classes.items():
                    class_id = int(class_id)
                    self._algo_classes[algo_id][class_id] = {
                        'name': class_info.get('name', f'class_{class_id}'),
                        'name_cn': class_info.get('name_cn', class_info.get('name', f'类别{class_id}'))
                    }
            
            # 统计各模型类型数量
            model_type_counts = {}
            for mt in self._algo_model_types.values():
                model_type_counts[mt] = model_type_counts.get(mt, 0) + 1
            
            print(f"[配置] 已从 {self.config_path} 加载 {len(self._algo_names)} 个算法配置")
            print(f"[配置] 模型类型分布: {model_type_counts}")
            
        except Exception as e:
            print(f"[错误] 加载算法类别配置失败: {e}")
            print("[警告] 将使用内置默认配置")
            self._use_default_config()
    
    def _use_default_config(self):
        """使用内置默认配置（兜底方案）"""
        # 内置的默认配置
        default_algo_classes = {
            1: {0: ("dead", "死亡"), 1: ("heavy", "重度患病"), 2: ("light", "轻度患病")},
            2: {0: ("waterpollution", "水污染"), 1: ("floatingdebris", "漂浮碎片"), 2: ("abandonedships", "废弃船只"), 3: ("fishingandaquaculture", "渔业和水产养殖"), 4: ("waste", "垃圾")},
            3: {0: ("bottle", "瓶子"), 1: ("grass", "草"), 2: ("branch", "树枝"), 3: ("milk-box", "牛奶盒"), 4: ("plastic-bag", "塑料袋"), 5: ("plastic-garbage", "塑料垃圾袋"), 6: ("ball", "球"), 7: ("leaf", "叶子")},
            4: {0: ("ignored", "忽略"), 1: ("swimmer", "游泳者"), 2: ("boat", "船"), 3: ("jetski", "水上摩托艇"), 4: ("life_saving_appliances", "救生设备"), 5: ("buoy", "浮标")},
            5: {0: ("license_plate", "车牌")},
            6: {0: ("vehicle", "车辆")},
            7: {0: ("Alligator Crack", "龟裂"), 1: ("Longitudinal Crack", "纵向裂缝"), 2: ("Longitudinal Patch", "纵向修补块"), 3: ("Manhole Cover", "检查井井盖"), 4: ("Pothole", "坑洞"), 5: ("Transverse Crack", "横向裂缝"), 6: ("Transverse Patch", "横向修补块")},
            8: {0: ("BAD_BILLBOARD", "违规广告牌"), 1: ("BROKEN_SIGNAGE", "破损标识牌"), 2: ("CLUTTER_SIDEWALK", "人行道杂物堆积"), 3: ("CONSTRUCTION_ROAD", "施工路段"), 4: ("FADED_SIGNAGE", "褪色标识牌"), 5: ("GARBAGE", "垃圾堆积"), 6: ("GRAFFITI", "涂鸦乱画"), 7: ("POTHOLES", "路面坑洞"), 8: ("SAND_ON_ROAD", "路面积沙"), 9: ("UNKEPT_FACADE", "建筑外立面破损")},
            9: {0: ("car", "车"), 1: ("people", "人")},
            10: {0: ("ShuiBianDiaoYu", "水边钓鱼"), 1: ("YouYongNiShui", "游泳溺水"), 2: ("DiaoYuSan", "钓鱼伞"), 3: ("boat", "船")},
            11: {0: ("crane", "起重机"), 1: ("excavator", "挖掘机"), 2: ("tractor", "拖拉机"), 3: ("truck", "卡车")},
            12: {0: ("straw", "秸秆堆")},
            13: {0: ("no_change", "无变化"), 1: ("change", "变化区域")},
            14: {0: ("zdjy", "占道经营")},
            15: {0: ("Bench", "长椅"), 1: ("Commercial_Trash", "商业垃圾"), 2: ("Dumping-sites", "非法倾倒点"), 3: ("Green_Land", "绿地"), 4: ("Hole", "孔洞"), 5: ("Jersey_Barrier", "泽西护栏"), 6: ("Land", "地块"), 7: ("Raw_Material", "原材料"), 8: ("Trash", "生活垃圾")},
            16: {0: ("trash", "垃圾"), 1: ("bare_soil", "裸土")},
            17: {0: ("blue_canopy", "蓝色天篷"), 1: ("others", "其他违建"), 2: ("green_shack", "改装绿色小屋")},
            18: {0: ("smoke", "烟雾"), 1: ("fire", "火")},
            19: {0: ("defected-pv-cells", "有缺陷的光伏电池")},
            20: {0: ("person", "人"), 1: ("car", "车"), 2: ("bicycle", "自行车")},
            21: {0: ("wall_corrosion", "墙体腐蚀"), 1: ("wall_crack", "墙体开裂"), 2: ("wall_deterioration", "墙体劣化"), 3: ("wall_mold", "墙模"), 4: ("wall_stain", "墙面污渍")},
            22: {0: ("poppy-opium", "罂粟")},
            23: {0: ("Lodged", "作物倒伏")},
            24: {0: ("backhoe_loader", "反铲装载机"), 1: ("compactor", "压路机"), 2: ("concrete_mixer_truck", "混凝土搅拌车"), 3: ("dozer", "推土机"), 4: ("dump_truck", "倾卸卡车"), 5: ("excavator", "挖掘机"), 6: ("grader", "平地机"), 7: ("helmet", "安全头盔"), 8: ("mobile_crane", "移动式起重机"), 9: ("person", "人"), 10: ("tower_crane", "塔式起重机"), 11: ("vest", "背心"), 12: ("wheel_loader", "轮式装载机")},
            999: {0: ("face", "人脸")},
        }
        
        default_algo_names = {
            1: "松线虫害识别", 2: "河道淤积识别", 3: "漂浮物识别", 4: "游泳涉水识别",
            5: "车牌识别", 6: "交通拥堵识别", 7: "路面破损识别", 8: "路面污染",
            9: "人群聚集识别", 10: "非法垂钓识别", 11: "施工识别", 12: "秸秆焚烧",
            13: "变化检测", 14: "占道经营识别", 15: "垃圾堆放识别", 16: "裸土未覆盖识别",
            17: "建控区违建识别", 18: "烟火识别", 19: "光伏板缺陷检测", 20: "园区夜间入侵检测",
            21: "园区外立面病害识别", 22: "罂粟识别", 23: "作物倒伏检测", 24: "林业侵占",
            999: "人脸检测"
        }
        
        # 默认模型类型配置（更新为最新）
        default_model_types = {algo_id: 'yolov5' for algo_id in default_algo_names.keys()}
        # yolov11 (640)
        default_model_types[6] = 'yolov11'
        # yolov11_720 (736)
        default_model_types[7] = 'yolov11_720'
        default_model_types[8] = 'yolov11_720'
        default_model_types[10] = 'yolov11_720'
        default_model_types[12] = 'yolov11_720'
        default_model_types[14] = 'yolov11_720'
        default_model_types[15] = 'yolov11_720'
        default_model_types[16] = 'yolov11_720'
        default_model_types[18] = 'yolov11_720'
        # change_detection (256)
        default_model_types[13] = 'change_detection'
        
        self._algo_names = default_algo_names
        self._algo_model_types = default_model_types
        for algo_id, classes in default_algo_classes.items():
            self._algo_classes[algo_id] = {}
            for class_id, (name, name_cn) in classes.items():
                self._algo_classes[algo_id][class_id] = {'name': name, 'name_cn': name_cn}
        
        print(f"[配置] 使用内置默认配置，共 {len(self._algo_names)} 个算法")
    
    def get_model_type(self, algo_id: int) -> str:
        """
        获取算法的模型类型
        
        Args:
            algo_id: 算法ID
        
        Returns:
            模型类型: 'yolov5' | 'yolov11' | 'yolov11_720' | 'change_detection'
            默认返回 'yolov5'
        """
        return self._algo_model_types.get(algo_id, 'yolov5')
    
    def get_input_size(self, algo_id: int) -> int:
        """
        获取算法的输入尺寸
        
        Args:
            algo_id: 算法ID
        
        Returns:
            输入尺寸: 640 | 736 | 256
        """
        model_type = self.get_model_type(algo_id)
        return MODEL_INPUT_SIZES.get(model_type, 640)
    
    def get_algo_name(self, algo_id: int) -> str:
        """获取算法名称"""
        return self._algo_names.get(algo_id, f"算法{algo_id}")
    
    def get_class_name(self, algo_id: int, class_id: int) -> str:
        """
        获取指定算法的类别英文名
        
        Args:
            algo_id: 算法ID
            class_id: 类别ID
        
        Returns:
            类别英文名，如果不存在返回 "class_{class_id}"
        """
        if algo_id in self._algo_classes and class_id in self._algo_classes[algo_id]:
            return self._algo_classes[algo_id][class_id]['name']
        return f"class_{class_id}"
    
    def get_class_name_cn(self, algo_id: int, class_id: int) -> str:
        """
        获取指定算法的类别中文名
        
        Args:
            algo_id: 算法ID
            class_id: 类别ID
        
        Returns:
            类别中文名，如果不存在返回英文名或 "类别{class_id}"
        """
        if algo_id in self._algo_classes and class_id in self._algo_classes[algo_id]:
            return self._algo_classes[algo_id][class_id]['name_cn']
        return self.get_class_name(algo_id, class_id)
    
    def get_class_info(self, algo_id: int, class_id: int) -> Dict[str, str]:
        """
        获取指定算法的类别完整信息
        
        Args:
            algo_id: 算法ID
            class_id: 类别ID
        
        Returns:
            包含 name 和 name_cn 的字典
        """
        if algo_id in self._algo_classes and class_id in self._algo_classes[algo_id]:
            return self._algo_classes[algo_id][class_id].copy()
        return {'name': f"class_{class_id}", 'name_cn': f"类别{class_id}"}
    
    def get_all_algo_ids(self) -> List[int]:
        """获取所有算法ID列表"""
        return sorted(self._algo_names.keys())
    
    def has_algo(self, algo_id: int) -> bool:
        """检查算法是否存在"""
        return algo_id in self._algo_names
    
    def get_algo_classes(self, algo_id: int) -> Dict[int, Dict[str, str]]:
        """获取指定算法的所有类别"""
        return self._algo_classes.get(algo_id, {}).copy()


# 全局配置实例（延迟初始化）
_algo_config: AlgoClassesConfig = None


def get_algo_config() -> AlgoClassesConfig:
    """获取算法配置实例（单例模式）"""
    global _algo_config
    if _algo_config is None:
        _algo_config = AlgoClassesConfig(ALGO_CLASSES_CONFIG)
    return _algo_config


# 兼容性别名：保留 ALGO_MAP 供外部引用
def get_algo_map() -> Dict[int, str]:
    """获取算法ID到名称的映射（兼容旧代码）"""
    config = get_algo_config()
    return {algo_id: config.get_algo_name(algo_id) for algo_id in config.get_all_algo_ids()}

# ACL 常量
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# 全局 ACL 初始化标志
_acl_initialized = False
_acl_init_lock = threading.Lock()

# 变化检测模型配置
CHANGE_DETECTION_INPUT_W = 256
CHANGE_DETECTION_INPUT_H = 256


def init_acl_once():
    """全局 ACL 初始化（仅一次）"""
    global _acl_initialized
    with _acl_init_lock:
        if not _acl_initialized:
            ret = acl.init()
            if ret != 0:
                raise RuntimeError(f"acl.init() failed: {ret}")
            _acl_initialized = True
            print(f"[ACL] 全局初始化成功")


# ============================================================
# NPU 设备池（轮询负载均衡）
# ============================================================
class NPUDevicePool:
    """NPU 设备池 - 轮询负载均衡"""

    def __init__(self, device_ids: List[int] = None, active_count: int = None):
        """
        Args:
            device_ids: 可用的设备 ID 列表
            active_count: 实际用于轮询的设备数量（用于模型副本数限制）
        """
        if device_ids is None:
            device_ids = list(range(16))
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        # 实际用于轮询的设备数量
        self.active_count = min(active_count or self.num_devices, self.num_devices)
        self.active_device_ids = device_ids[:self.active_count]
        
        self._robin_index = 0
        self._robin_lock = threading.Lock()
        self._device_usage = defaultdict(int)
        self._stats_lock = threading.Lock()
        print(f"[NPU Pool] 初始化完成，总设备数: {self.num_devices}, 活跃设备数: {self.active_count}")
        print(f"[NPU Pool] 活跃设备: {self.active_device_ids}")

    def acquire(self) -> int:
        """获取一个设备ID（在活跃设备范围内轮询）"""
        with self._robin_lock:
            device_id = self.active_device_ids[self._robin_index]
            self._robin_index = (self._robin_index + 1) % self.active_count
        with self._stats_lock:
            self._device_usage[device_id] += 1
        return device_id

    def get_stats(self) -> dict:
        with self._stats_lock:
            return {
                "total_devices": self.num_devices,
                "active_devices": self.active_count,
                "device_ids": self.device_ids,
                "active_device_ids": self.active_device_ids,
                "usage_count": dict(self._device_usage),
                "total_requests": sum(self._device_usage.values())
            }

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'=' * 60}")
        print(f"NPU 设备池统计")
        print(f"{'=' * 60}")
        print(f"总设备数: {stats['total_devices']}, 活跃设备数: {stats['active_devices']}")
        print(f"请求总数: {stats['total_requests']}")
        print(f"\n各设备使用情况:")
        for dev_id in sorted(stats['usage_count'].keys()):
            count = stats['usage_count'][dev_id]
            pct = (count / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            print(f"  Device {dev_id:2d}: {count:5d} 次 ({pct:5.1f}%)")
        print(f"{'=' * 60}\n")


# ============================================================
# 变化检测模型推理器
# ============================================================
class ChangeDetectionOMDetector:
    """变化检测 OM 模型推理器"""

    def __init__(self, model_path: str, device_id: int, algo_id: int = 13):
        self.model_path = model_path
        self.device_id = device_id
        self.algo_id = algo_id
        self.input_w = CHANGE_DETECTION_INPUT_W
        self.input_h = CHANGE_DETECTION_INPUT_H

        ret = acl.rt.set_device(device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_device({device_id}) failed: {ret}")

        self.context, ret = acl.rt.create_context(device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_context({device_id}) failed: {ret}")

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.load_from_file({model_path}) failed: {ret}")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_desc() failed: {ret}")

        print(f"[ChangeDetection] 模型加载成功: device={device_id}, path={model_path}")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """图像预处理"""
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return np.ascontiguousarray(img)

    def detect(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        变化检测推理
        
        Returns:
            (mask_img, change_ratio, infer_time)
        """
        acl.rt.set_context(self.context)
        
        orig_h, orig_w = image1.shape[:2]
        
        data1 = self.preprocess(image1)
        data2 = self.preprocess(image2)
        
        t0 = time.time()
        change_mask = self._infer(data1, data2)
        infer_time = time.time() - t0
        
        mask_resized = cv2.resize(change_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_img = (mask_resized * 255).astype(np.uint8)
        change_ratio = np.sum(mask_resized == 1) / (orig_h * orig_w)
        
        return mask_img, change_ratio, infer_time

    def _infer(self, img1_data: np.ndarray, img2_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        acl.rt.set_context(self.context)
        
        input_dataset = acl.mdl.create_dataset()
        input_buffers = []
        
        for img_data in [img1_data, img2_data]:
            input_size = img_data.nbytes
            input_buffer, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc input failed: {ret}")
            
            input_bytes = img_data.tobytes()
            input_ptr = acl.util.bytes_to_ptr(input_bytes)
            ret = acl.rt.memcpy(input_buffer, input_size, input_ptr, len(input_bytes), ACL_MEMCPY_HOST_TO_DEVICE)
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy input failed: {ret}")
            
            data_buf = acl.create_data_buffer(input_buffer, input_size)
            acl.mdl.add_dataset_buffer(input_dataset, data_buf)
            input_buffers.append((input_buffer, data_buf))
        
        output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)
        output_buffer, ret = acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc output failed: {ret}")
        
        output_data_buf = acl.create_data_buffer(output_buffer, output_size)
        acl.mdl.add_dataset_buffer(output_dataset, output_data_buf)
        
        try:
            ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
            if ret != 0:
                raise RuntimeError(f"acl.mdl.execute failed: {ret}")
            
            host_output = np.zeros((1, 2, self.input_h, self.input_w), dtype=np.float32)
            host_bytes = host_output.tobytes()
            host_ptr = acl.util.bytes_to_ptr(host_bytes)
            
            ret = acl.rt.memcpy(host_ptr, len(host_bytes), output_buffer, output_size, ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy output failed: {ret}")
            
            host_output = np.frombuffer(host_bytes, dtype=np.float32).reshape(1, 2, self.input_h, self.input_w)
            change_mask = np.argmax(host_output[0], axis=0).astype(np.uint8)
            
            return change_mask
            
        finally:
            for input_buf, data_buf in input_buffers:
                acl.destroy_data_buffer(data_buf)
                acl.rt.free(input_buf)
            acl.destroy_data_buffer(output_data_buf)
            acl.rt.free(output_buffer)
            acl.mdl.destroy_dataset(input_dataset)
            acl.mdl.destroy_dataset(output_dataset)

    def __del__(self):
        try:
            if hasattr(self, 'model_desc'):
                acl.mdl.destroy_desc(self.model_desc)
            if hasattr(self, 'model_id'):
                acl.mdl.unload(self.model_id)
            if hasattr(self, 'context'):
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
        except:
            pass


# ============================================================
# YOLO OM 模型推理器（兼容 YOLOv5、YOLOv11、YOLOv11_720）
# ============================================================
class YOLOOMDetector:
    """
    YOLO OM 模型推理器
    
    根据配置文件中的 model_type 字段选择解析方式：
    - yolov5: 输出形状 [25200, 5+num_classes]，包含 objectness，输入 640
    - yolov11: 输出形状 [4+num_classes, 8400]，无 objectness，输入 640
    - yolov11_720: 输出形状 [4+num_classes, 11109]，无 objectness，输入 736
    """

    def __init__(self, model_path: str, device_id: int, algo_id: int):
        self.model_path = model_path
        self.device_id = device_id
        self.algo_id = algo_id
        
        # 从配置文件获取模型类型和输入尺寸
        algo_config = get_algo_config()
        self._config_model_type = algo_config.get_model_type(algo_id)
        self._input_size = algo_config.get_input_size(algo_id)
        self._model_type = None  # 完整的模型类型信息，初始化时设置

        ret = acl.rt.set_device(device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_device({device_id}) failed: {ret}")

        self.context, ret = acl.rt.create_context(device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_context({device_id}) failed: {ret}")

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.load_from_file({model_path}) failed: {ret}")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_desc() failed: {ret}")

        # 获取输出大小
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)
        
        # 直接从模型获取输出维度信息，设置正确的解析参数
        self._init_model_type()
        
        print(f"[YOLO] 模型加载成功: algo={algo_id}, device={device_id}, "
              f"model_type={self._config_model_type}, input_size={self._input_size}, "
              f"output_bytes={self.output_size}, num_preds={self._model_type['num_preds']}, "
              f"num_cols={self._model_type['num_cols']}")
    
    def _init_model_type(self):
        """
        从模型描述中获取输出维度，初始化模型类型信息
        
        YOLOv11 输出维度: [1, num_cols, num_preds] 如 [1, 11, 11109]
          - num_cols 较小 (4~100), num_preds 较大 (8400, 11109等)
          - 需要 reshape(num_cols, num_preds).T 得到 [num_preds, num_cols]
          
        YOLOv5 输出维度: [1, num_preds, num_cols] 如 [1, 25200, 12]
          - num_preds 较大 (25200等), num_cols 较小 (5~100)
          - 直接 reshape(num_preds, num_cols)
        """
        config_is_yolov11 = self._config_model_type in ('yolov11', 'yolov11_720')
        
        num_preds = None
        num_cols = None
        actual_is_yolov11 = None  # 根据实际模型输出判断
        
        # 尝试从模型获取输出维度
        try:
            dims_info = acl.mdl.get_output_dims(self.model_desc, 0)
            if dims_info and isinstance(dims_info, tuple) and len(dims_info) > 0:
                dims_dict = dims_info[0]
                if isinstance(dims_dict, dict) and 'dims' in dims_dict:
                    dims = dims_dict['dims']
                    dim_count = dims_dict.get('dimCount', len(dims))
                    
                    if dim_count == 3 and len(dims) >= 3:
                        dim1 = dims[1]
                        dim2 = dims[2]
                        
                        # 根据维度大小判断模型类型
                        # YOLOv11: dim1 小 (num_cols), dim2 大 (num_preds)
                        # YOLOv5:  dim1 大 (num_preds), dim2 小 (num_cols)
                        if dim1 < dim2:
                            # YOLOv11 格式: [1, num_cols, num_preds]
                            num_cols = dim1
                            num_preds = dim2
                            actual_is_yolov11 = True
                        else:
                            # YOLOv5 格式: [1, num_preds, num_cols]
                            num_preds = dim1
                            num_cols = dim2
                            actual_is_yolov11 = False
                            
        except Exception as e:
            print(f"[警告] algo={self.algo_id} 无法从模型获取输出维度: {e}")
        
        # 如果无法从模型获取，则根据输出大小计算
        if num_preds is None or num_cols is None:
            total_elements = self.output_size // 2  # FP16
            
            if config_is_yolov11:
                # 尝试常见的 predictions 数量
                for expected_preds in [11109, 8400, 33600]:
                    if total_elements % expected_preds == 0:
                        num_preds = expected_preds
                        num_cols = total_elements // expected_preds
                        actual_is_yolov11 = True
                        break
                if num_preds is None:
                    num_preds = 11109 if self._input_size == 736 else 8400
                    num_cols = total_elements // num_preds
                    actual_is_yolov11 = True
            else:
                # YOLOv5
                for expected_preds in [25200, 100800]:
                    if total_elements % expected_preds == 0:
                        num_preds = expected_preds
                        num_cols = total_elements // expected_preds
                        actual_is_yolov11 = False
                        break
                if num_preds is None:
                    num_preds = 25200
                    num_cols = total_elements // num_preds
                    actual_is_yolov11 = False
        
        # 如果配置与实际不匹配，打印警告
        if actual_is_yolov11 is not None and actual_is_yolov11 != config_is_yolov11:
            actual_type = 'YOLOv11' if actual_is_yolov11 else 'YOLOv5'
            config_type = 'YOLOv11' if config_is_yolov11 else 'YOLOv5'
            print(f"[警告] algo={self.algo_id} 配置为 {config_type}，但实际模型是 {actual_type}，将按实际类型处理")
        
        # 使用实际检测到的模型类型
        is_yolov11 = actual_is_yolov11 if actual_is_yolov11 is not None else config_is_yolov11
        
        # 设置模型类型信息
        if is_yolov11:
            self._model_type = {
                'name': f'YOLOv11({self._input_size})',
                'has_objectness': False,
                'transpose': True,  # YOLOv11 需要转置
                'num_preds': num_preds,
                'num_cols': num_cols
            }
        else:
            self._model_type = {
                'name': 'YOLOv5',
                'has_objectness': True,
                'transpose': False,  # YOLOv5 不需要转置
                'num_preds': num_preds,
                'num_cols': num_cols
            }

    def preprocess(self, image: np.ndarray, input_size: int = None) -> Tuple[np.ndarray, tuple]:
        """图像预处理（letterbox）"""
        if input_size is None:
            input_size = self._input_size
            
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
        img_batch = np.expand_dims(img_norm, axis=0)
        return np.ascontiguousarray(img_batch), (orig_h, orig_w, r, top, left)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """执行模型推理"""
        acl.rt.set_context(self.context)
        input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)

        input_buffer, _ = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
        output_buffer, _ = acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST)

        try:
            input_bytes = input_data.tobytes()
            input_ptr = acl.util.bytes_to_ptr(input_bytes)
            acl.rt.memcpy(input_buffer, input_size, input_ptr, len(input_bytes), ACL_MEMCPY_HOST_TO_DEVICE)

            input_dataset = acl.mdl.create_dataset()
            output_dataset = acl.mdl.create_dataset()
            input_data_buffer = acl.create_data_buffer(input_buffer, input_size)
            output_data_buffer = acl.create_data_buffer(output_buffer, output_size)

            acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)
            acl.mdl.add_dataset_buffer(output_dataset, output_data_buffer)
            acl.mdl.execute(self.model_id, input_dataset, output_dataset)

            output_fp16 = np.zeros(output_size // 2, dtype=np.float16)
            output_bytes = output_fp16.tobytes()
            output_ptr = acl.util.bytes_to_ptr(output_bytes)
            acl.rt.memcpy(output_ptr, len(output_bytes), output_buffer, output_size, ACL_MEMCPY_DEVICE_TO_HOST)
            output_fp16 = np.frombuffer(output_bytes, dtype=np.float16)

            acl.destroy_data_buffer(input_data_buffer)
            acl.destroy_data_buffer(output_data_buffer)
            acl.mdl.destroy_dataset(input_dataset)
            acl.mdl.destroy_dataset(output_dataset)

            return output_fp16.astype(np.float32)
        finally:
            acl.rt.free(input_buffer)
            acl.rt.free(output_buffer)

    def detect(self, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[Dict[str, Any]]:
        """执行目标检测"""
        acl.rt.set_context(self.context)
        input_data, img_info = self.preprocess(image, self._input_size)
        output = self.infer(input_data)
        return self._postprocess(output, img_info, conf_thres, iou_thres)

    def _postprocess(self, output: np.ndarray, img_info: tuple, conf_thres: float, iou_thres: float) -> List[Dict[str, Any]]:
        """
        后处理：根据初始化时确定的 model_type 进行解析
        """
        orig_h, orig_w, ratio, pad_top, pad_left = img_info
        
        num_preds = self._model_type['num_preds']
        num_cols = self._model_type['num_cols']
        has_objectness = self._model_type['has_objectness']
        need_transpose = self._model_type['transpose']
        
        # 重塑输出张量
        if need_transpose:
            # YOLOv11 格式: [num_cols, num_preds] -> 转置为 [num_preds, num_cols]
            predictions = output.reshape(num_cols, num_preds).T
        else:
            # YOLOv5 格式: [num_preds, num_cols]
            predictions = output.reshape(num_preds, num_cols)
        
        # YOLOv5 至少需要5列 (x,y,w,h,obj)，YOLOv11 至少需要4列 (x,y,w,h)
        min_cols = 5 if has_objectness else 4
        if num_cols < min_cols:
            return []

        # 提取边界框 (x_center, y_center, width, height)
        boxes = predictions[:, :4].copy()
        
        if has_objectness:
            # YOLOv5: 有 objectness 置信度
            obj_conf = predictions[:, 4]
            num_classes = num_cols - 5
            
            if num_classes > 0:
                class_scores = predictions[:, 5:5+num_classes]
                class_ids = np.argmax(class_scores, axis=1)
                class_conf = np.max(class_scores, axis=1)
                scores = obj_conf * class_conf  # 最终置信度 = obj_conf × class_conf
            else:
                scores = obj_conf
                class_ids = np.zeros(len(scores), dtype=int)
        else:
            # YOLOv11: 无 objectness，直接使用类别置信度
            num_classes = num_cols - 4
            
            if num_classes > 0:
                class_scores = predictions[:, 4:4+num_classes]
                # YOLOv11 输出已经是 sigmoid 后的值（0-1范围），直接使用
                class_ids = np.argmax(class_scores, axis=1)
                scores = np.max(class_scores, axis=1)
            else:
                return []

        # 置信度过滤
        mask = scores > conf_thres
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return []

        # 坐标转换 xywh -> xyxy
        boxes = self._xywh2xyxy(boxes)
        
        # 去除 padding 偏移
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        
        # 缩放回原始图像尺寸
        boxes /= ratio

        # 过滤无效框
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]

        if len(boxes) == 0:
            return []

        # NMS
        keep = self._nms(boxes, scores, iou_thres)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        
        # 裁剪到图像边界
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # 构建检测结果
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                class_name = self._get_class_name(int(cls_id))
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "class_id": int(cls_id),
                    "class_name": class_name
                })
        return detections

    def _get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        config = get_algo_config()
        return config.get_class_name(self.algo_id, class_id)

    @staticmethod
    def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        """中心点格式转角点格式"""
        boxes_xyxy = np.copy(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes_xyxy

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

    def __del__(self):
        try:
            if hasattr(self, 'model_desc'):
                acl.mdl.destroy_desc(self.model_desc)
            if hasattr(self, 'model_id'):
                acl.mdl.unload(self.model_id)
            if hasattr(self, 'context'):
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
        except:
            pass


# 兼容性别名：保持向后兼容
YOLOv5OMDetector = YOLOOMDetector


# ============================================================
# 模型管理器
# ============================================================
class OMModelManager:
    """
    模型管理器
    
    支持扫描两种模型文件：
    - {algo_id}_bs1.om: 标准模型 (640x640)
    - {algo_id}_720.om: 720模型 (736x736)
    
    根据算法配置的 model_type 选择对应的模型文件
    """
    
    def __init__(self, model_dir: str, device_pool: NPUDevicePool, preload: bool = True):
        self.model_dir = model_dir
        self.device_pool = device_pool
        self.preload = preload
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._load_count = 0
        self._hit_count = 0
        self._scan_models()
        if self.preload:
            self._preload_all_models()
        print(f"[模型管理器] 初始化完成")

    def _scan_models(self):
        """扫描模型文件，根据算法配置选择合适的模型"""
        self.model_files = {}
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        
        algo_config = get_algo_config()
        all_files = os.listdir(self.model_dir)
        
        # 分类模型文件
        bs1_models = {}  # {algo_id: path}
        m720_models = {}  # {algo_id: path}
        
        for filename in all_files:
            if not filename.endswith('.om'):
                continue
            
            filepath = os.path.join(self.model_dir, filename)
            
            # 解析 720 模型: {algo_id}_720.om
            if '_720.om' in filename:
                base_name = filename.replace('_720.om', '')
                try:
                    algo_id = int(base_name)
                    m720_models[algo_id] = filepath
                except ValueError:
                    continue
            # 解析标准模型: {algo_id}_bs1.om
            elif '_bs1.om' in filename:
                base_name = filename.replace('_bs1.om', '')
                try:
                    algo_id = int(base_name)
                    bs1_models[algo_id] = filepath
                except ValueError:
                    if base_name == "人脸检测":
                        bs1_models[999] = filepath
                    continue
        
        # 打印扫描到的模型
        print(f"[模型扫描] bs1 模型: {sorted(bs1_models.keys())}")
        print(f"[模型扫描] 720 模型: {sorted(m720_models.keys())}")
        
        # 根据算法配置选择模型
        all_algo_ids = set(bs1_models.keys()) | set(m720_models.keys())
        
        for algo_id in all_algo_ids:
            model_type = algo_config.get_model_type(algo_id)
            
            # yolov11_720 优先使用 720 模型
            if model_type == 'yolov11_720' and algo_id in m720_models:
                self.model_files[algo_id] = m720_models[algo_id]
            # 其他类型使用 bs1 模型
            elif algo_id in bs1_models:
                self.model_files[algo_id] = bs1_models[algo_id]
            # 如果没有 bs1 模型，尝试使用 720 模型
            elif algo_id in m720_models:
                self.model_files[algo_id] = m720_models[algo_id]
        
        # 打印扫描结果
        print(f"[模型管理器] 发现 {len(self.model_files)} 个模型:")
        for algo_id in sorted(self.model_files.keys()):
            model_path = self.model_files[algo_id]
            model_type = algo_config.get_model_type(algo_id)
            input_size = algo_config.get_input_size(algo_id)
            print(f"  算法 {algo_id:3d}: {os.path.basename(model_path):<20s} "
                  f"(type={model_type}, input={input_size})")

    def _preload_all_models(self):
        # 确定要预加载到哪些设备
        preload_device_ids = self.device_pool.device_ids[:MODEL_REPLICAS]
        
        print(f"\n{'=' * 70}")
        print(f"开始预加载: {len(self.model_files)} 个模型 × {len(preload_device_ids)} 个设备")
        print(f"预加载设备: {preload_device_ids}")
        print(f"{'=' * 70}\n")
        start_time = time.time()
        total = len(self.model_files) * len(preload_device_ids)
        count = 0

        for device_id in preload_device_ids:
            print(f"\n[Device {device_id}] 加载中...")
            for algo_id in sorted(self.model_files.keys()):
                model_path = self.model_files[algo_id]
                cache_key = (algo_id, device_id)
                try:
                    algo_config = get_algo_config()
                    model_type = algo_config.get_model_type(algo_id)
                    
                    if model_type == 'change_detection':
                        detector = ChangeDetectionOMDetector(model_path, device_id, algo_id)
                    else:
                        # 使用兼容 YOLOv5/v11/v11_720 的检测器
                        detector = YOLOOMDetector(model_path, device_id, algo_id)
                    self._cache[cache_key] = detector
                    count += 1
                    if count % 10 == 0 or count == total:
                        progress = count / total * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / count) * (total - count) if count > 0 else 0
                        print(f"  进度: {count}/{total} ({progress:.1f}%) | 已用时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s")
                except Exception as e:
                    print(f"  ❌ 加载失败: algo={algo_id}, device={device_id}, 错误: {e}")

        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"✅ 预加载完成！加载数量: {len(self._cache)}/{total}, 耗时: {elapsed:.1f}s")
        print(f"{'=' * 70}\n")
        self._load_count = len(self._cache)

    def get_detector(self, algo_id: int, device_id: Optional[int] = None):
        if algo_id not in self.model_files:
            raise ValueError(f"模型不存在: algo_id={algo_id}")
        if device_id is None:
            device_id = self.device_pool.acquire()
        cache_key = (algo_id, device_id)
        with self._cache_lock:
            if cache_key in self._cache:
                self._hit_count += 1
                return self._cache[cache_key]
            if self.preload:
                raise RuntimeError(f"预加载模式下缓存未命中: {cache_key}")
            model_path = self.model_files[algo_id]
            algo_config = get_algo_config()
            model_type = algo_config.get_model_type(algo_id)
            
            if model_type == 'change_detection':
                detector = ChangeDetectionOMDetector(model_path, device_id, algo_id)
            else:
                detector = YOLOOMDetector(model_path, device_id, algo_id)
            self._cache[cache_key] = detector
            self._load_count += 1
            return detector

    def get_stats(self) -> dict:
        with self._cache_lock:
            total_ops = self._load_count + self._hit_count
            hit_rate = (self._hit_count / total_ops * 100) if total_ops > 0 else 0
            return {
                "total_models": len(self.model_files),
                "cached_instances": len(self._cache),
                "preload_enabled": self.preload,
                "load_count": self._load_count,
                "hit_count": self._hit_count,
                "hit_rate": f"{hit_rate:.1f}%"
            }

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'=' * 60}")
        print(f"模型管理器统计")
        print(f"{'=' * 60}")
        print(f"可用模型数: {stats['total_models']}, 已加载实例: {stats['cached_instances']}")
        print(f"命中率: {stats['hit_rate']}")
        print(f"{'=' * 60}\n")


# ============================================================
# 图像处理函数
# ============================================================
def decode_base64_image(b64_str: str) -> np.ndarray:
    """解码 base64 图像"""
    try:
        img_bytes = base64.b64decode(b64_str, validate=True)
    except Exception as e:
        raise ValueError(f"base64 解码失败: {e}")
    
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("图像解码失败")
    
    h, w = img.shape[:2]
    if max(w, h) > MAX_DECODE_SIDE:
        scale = MAX_DECODE_SIDE / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img


def encode_image_base64(img: np.ndarray) -> str:
    """将图像编码为 base64 PNG"""
    success, encoded = cv2.imencode('.png', img)
    if not success:
        raise ValueError("图像编码失败")
    return base64.b64encode(encoded.tobytes()).decode('utf-8')


def bbox_to_normalized(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    nx1, ny1 = x1 / img_w, y1 / img_h
    nx2, ny2 = x2 / img_w, y2 / img_h
    return [round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)]


# ============================================================
# gRPC 服务
# ============================================================
_device_pool: NPUDevicePool = None
_model_manager: OMModelManager = None
_plate_recognizer = None


class DetectionServicer(detection_pb2_grpc.DetectionServiceServicer):
    """NPU 检测服务"""

    def Detect(self, request, context):
        """通用检测接口（目标检测类算法）"""
        metadata = dict(context.invocation_metadata())
        api_key = metadata.get('x-api-key', '')
        if api_key != API_KEY:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details('未授权: API Key 无效')
            return detection_pb2.DetectResponse()

        algo_config = get_algo_config()
        if not algo_config.has_algo(request.algorithm_id):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f'算法 ID {request.algorithm_id} 不存在')
            return detection_pb2.DetectResponse()

        # 算法13是变化检测，需要使用 DetectChange 接口
        if request.algorithm_id == 13:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('算法 ID 13 (变化检测) 请使用 DetectChange 接口')
            return detection_pb2.DetectResponse()

        try:
            img_np = decode_base64_image(request.image)
            img_h, img_w = img_np.shape[:2]

            detector = _model_manager.get_detector(request.algorithm_id)

            t0 = time.time()
            raw_detections = detector.detect(
                img_np,
                conf_thres=request.conf_threshold if request.conf_threshold > 0 else DEFAULT_CONF,
                iou_thres=0.45
            )
            detect_time = time.time() - t0

            detections = []
            for det in raw_detections:
                norm_bbox = bbox_to_normalized(det["bbox"], img_w, img_h)
                class_id = det.get("class_id", 0)
                # 使用算法配置获取类别名（按算法ID隔离，避免冲突）
                class_info = algo_config.get_class_info(request.algorithm_id, class_id)
                class_name = class_info['name']
                class_name_cn = class_info['name_cn']

                detection = detection_pb2.Detection(
                    class_id=class_id,
                    class_name=class_name,
                    class_name_cn=class_name_cn,
                    confidence=det["confidence"],
                    bbox=norm_bbox
                )
                
                if request.algorithm_id == 5 and _plate_recognizer:
                    try:
                        result = _plate_recognizer.recognize(img_np, det["bbox"])
                        if result["plate_number"]:
                            detection.plate_number = result["plate_number"]
                            detection.plate_type = result["plate_type"]
                            detection.plate_confidence = result["confidence"]
                    except:
                        pass
                
                detections.append(detection)

            data = detection_pb2.DetectionData(
                algorithm_id=request.algorithm_id,
                algorithm_name=algo_config.get_algo_name(request.algorithm_id),
                detections=detections,
                total_count=len(detections),
                detect_time=round(detect_time, 3)
            )

            return detection_pb2.DetectResponse(code=200, message="success", data=data)

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return detection_pb2.DetectResponse()
        except Exception as e:
            print(f"[错误] 检测失败: {e}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"内部错误: {e}")
            return detection_pb2.DetectResponse()

    def DetectChange(self, request, context):
        """
        变化检测接口（algorithm_id=13）
        
        输入: image1, image2 (base64编码的PNG图片)
        输出: mask (base64编码的PNG mask图片)
        """
        metadata = dict(context.invocation_metadata())
        api_key = metadata.get('x-api-key', '')
        if api_key != API_KEY:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details('未授权: API Key 无效')
            return detection_pb2.ChangeDetectResponse()

        if not request.image1:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('需要提供 image1 字段')
            return detection_pb2.ChangeDetectResponse()
        
        if not request.image2:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('需要提供 image2 字段')
            return detection_pb2.ChangeDetectResponse()

        try:
            # 解码base64图片
            img1_np = decode_base64_image(request.image1)
            img2_np = decode_base64_image(request.image2)
            
            img_h, img_w = img1_np.shape[:2]
            
            # 获取变化检测器
            detector = _model_manager.get_detector(13)
            
            # 执行推理
            mask_img, change_ratio, infer_time = detector.detect(img1_np, img2_np)
            
            # 将mask编码为base64
            mask_b64 = encode_image_base64(mask_img)
            
            return detection_pb2.ChangeDetectResponse(
                code=200,
                message="success",
                mask=mask_b64,
                width=img_w,
                height=img_h,
                change_ratio=round(change_ratio, 6),
                detect_time=round(infer_time, 3)
            )
            
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return detection_pb2.ChangeDetectResponse()
        except Exception as e:
            print(f"[错误] 变化检测失败: {e}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"内部错误: {e}")
            return detection_pb2.ChangeDetectResponse()

    def HealthCheck(self, request, context):
        device_stats = _device_pool.get_stats()
        model_stats = _model_manager.get_stats()
        return detection_pb2.HealthResponse(
            status="ok",
            device=f"NPU (Ascend 310P3) x{device_stats['total_devices']}",
            yolov5_repo="OM Models (YOLOv5/v11/v11_720 Compatible)",
            models_cached=model_stats['cached_instances'],
            image_backend="opencv",
            gpu_name=f"{device_stats['total_devices']} NPU devices",
            gpu_memory_allocated_mb=0.0,
            gpu_memory_reserved_mb=0.0
        )

    def GetVersion(self, request, context):
        algo_config = get_algo_config()
        return detection_pb2.VersionResponse(
            version="3.4.0-npu-unified",
            mode="offline-npu-preload",
            pytorch_version="N/A (OM Runtime)",
            opencv_version=cv2.__version__,
            device=f"Ascend NPU x{len(NPU_DEVICE_IDS)}",
            cuda_available=False,
            default_conf_threshold=DEFAULT_CONF,
            algo_supported=algo_config.get_all_algo_ids()
        )


# ============================================================
# 启动函数
# ============================================================
def _startup():
    global _device_pool, _model_manager, _plate_recognizer
    print("=" * 70)
    print("NPU 推理服务启动中...")
    print("支持模型: YOLOv5 / YOLOv8 / YOLOv11 / YOLOv11_720 (自动识别)")
    print("=" * 70)
    
    # 调整文件描述符限制（避免 File descriptor limit reached 错误）
    adjust_file_descriptor_limit(65535)
    
    # 初始化算法配置（从 YAML 文件加载）
    algo_config = get_algo_config()
    
    init_acl_once()
    # 传入 MODEL_REPLICAS 作为活跃设备数量
    _device_pool = NPUDevicePool(device_ids=NPU_DEVICE_IDS, active_count=MODEL_REPLICAS)
    _model_manager = OMModelManager(MODEL_DIR, _device_pool, preload=PRELOAD_ALL_MODELS)
    
    if _PLATE_RECOGNIZER_AVAILABLE:
        try:
            _plate_recognizer = UniversalPlateRecognizer(detect_level=1, rec_level=1)
        except:
            pass

    # 统计模型类型
    model_type_stats = {}
    for algo_id in _model_manager.model_files.keys():
        mt = algo_config.get_model_type(algo_id)
        model_type_stats[mt] = model_type_stats.get(mt, 0) + 1

    print("=" * 70)
    print(f"✅ 服务就绪")
    print(f"   配置文件: {ALGO_CLASSES_CONFIG}")
    print(f"   模型目录: {MODEL_DIR}")
    print(f"   NPU 设备: {len(NPU_DEVICE_IDS)} 个 (IDs: {NPU_DEVICE_IDS})")
    print(f"   gRPC 线程: {GRPC_WORKERS} 个")
    print(f"   预加载模型: {PRELOAD_ALL_MODELS}")
    print(f"   可用模型: {len(_model_manager.model_files)} 个")
    print(f"   模型类型: {model_type_stats}")
    print(f"   支持算法: {len(algo_config.get_all_algo_ids())} 个")
    print(f"   变化检测(algo=13): {'可用' if 13 in _model_manager.model_files else '未配置'}")
    print(f"   模型兼容: YOLOv5 / YOLOv8 / YOLOv11 / YOLOv11_720")
    print("=" * 70)


def serve():
    _startup()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=GRPC_WORKERS),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    detection_pb2_grpc.add_DetectionServiceServicer_to_server(DetectionServicer(), server)
    SERVICE_NAMES = (
        detection_pb2.DESCRIPTOR.services_by_name['DetectionService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:8000')
    server.start()
    print("\n🚀 gRPC 服务已启动，监听端口 8000\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[关闭] 服务停止中...")
        server.stop(0)


if __name__ == '__main__':
    serve()
