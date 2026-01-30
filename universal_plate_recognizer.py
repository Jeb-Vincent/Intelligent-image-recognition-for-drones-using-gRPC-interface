#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用车牌识别器 - 兼容多个HyperLPR3版本
"""

from typing import List, Dict, Any
import numpy as np
import cv2

# 尝试导入HyperLPR3（兼容多个版本）
HYPERLPR_AVAILABLE = False
HYPERLPR_VERSION = None

# 尝试方式1: LicensePlateCN (新版本)
try:
    from hyperlpr3 import LicensePlateCN
    HYPERLPR_AVAILABLE = True
    HYPERLPR_VERSION = "LicensePlateCN"
    print("[车牌识别] 使用 LicensePlateCN")
except ImportError:
    pass

# 尝试方式2: LicensePlateCatcher (旧版本)
if not HYPERLPR_AVAILABLE:
    try:
        from hyperlpr3 import LicensePlateCatcher
        HYPERLPR_AVAILABLE = True
        HYPERLPR_VERSION = "LicensePlateCatcher"
        print("[车牌识别] 使用 LicensePlateCatcher")
    except ImportError:
        pass

# 尝试方式3: HyperLPR3 类
if not HYPERLPR_AVAILABLE:
    try:
        from hyperlpr3 import HyperLPR3
        HYPERLPR_AVAILABLE = True
        HYPERLPR_VERSION = "HyperLPR3"
        print("[车牌识别] 使用 HyperLPR3")
    except ImportError:
        pass

if not HYPERLPR_AVAILABLE:
    print("[警告] HyperLPR3 未安装或导入失败")
    print("       安装方法: pip install hyperlpr3 --break-system-packages")


# 车牌类型映射
PLATE_TYPE_MAP = {
    "blue": "蓝牌",
    "yellow": "黄牌",
    "green": "绿牌新能源",
    "new_energy": "绿牌新能源",  # 别名
    "black": "黑牌",
    "white": "白牌",
    "yellow_green": "黄绿牌",
    "unknown": "未知类型",
    
    # 一些旧版本可能返回的类型
    "0": "蓝牌",
    "1": "黄牌",
    "2": "绿牌新能源",
}


class UniversalPlateRecognizer:
    """通用车牌识别器 - 兼容多个HyperLPR3版本"""
    
    def __init__(self, detect_level: int = 1, rec_level: int = 1):
        """
        初始化识别器
        
        Args:
            detect_level: 检测等级 (1-3)
            rec_level: 识别等级 (1-3)
        """
        self.recognizer = None
        self.version = HYPERLPR_VERSION
        self._predict_method = None  # 存储实际的预测方法
        
        if not HYPERLPR_AVAILABLE:
            print("[车牌识别] HyperLPR3 不可用")
            return
        
        try:
            if HYPERLPR_VERSION == "LicensePlateCN":
                from hyperlpr3 import LicensePlateCN
                self.recognizer = LicensePlateCN(
                    detect_level=detect_level,
                    rec_level=rec_level
                )
            elif HYPERLPR_VERSION == "LicensePlateCatcher":
                from hyperlpr3 import LicensePlateCatcher
                # 旧版本可能不支持level参数
                try:
                    self.recognizer = LicensePlateCatcher(
                        level=detect_level
                    )
                except TypeError:
                    self.recognizer = LicensePlateCatcher()
            elif HYPERLPR_VERSION == "HyperLPR3":
                from hyperlpr3 import HyperLPR3
                self.recognizer = HyperLPR3()
            
            # 🔧 检测实际可用的预测方法
            self._detect_predict_method()
            
            print(f"[车牌识别] {HYPERLPR_VERSION} 初始化成功, 预测方法: {self._predict_method}")
            
        except Exception as e:
            self.recognizer = None
            print(f"[车牌识别] 初始化失败: {e}")
    
    def _detect_predict_method(self):
        """检测识别器实际可用的预测方法"""
        if self.recognizer is None:
            return
        
        # 按优先级尝试不同的方法名
        method_names = ['predict', '__call__', 'recognize', 'detect', 'run', 'inference']
        
        for method_name in method_names:
            if method_name == '__call__':
                # 检查是否可调用
                if callable(self.recognizer):
                    self._predict_method = '__call__'
                    return
            elif hasattr(self.recognizer, method_name):
                method = getattr(self.recognizer, method_name)
                if callable(method):
                    self._predict_method = method_name
                    return
        
        # 打印所有可用方法，帮助调试
        available_methods = [m for m in dir(self.recognizer) if not m.startswith('_') and callable(getattr(self.recognizer, m, None))]
        print(f"[车牌识别] 警告: 未找到标准预测方法，可用方法: {available_methods}")
        
        # 如果找不到标准方法，尝试 __call__
        if callable(self.recognizer):
            self._predict_method = '__call__'
    
    def recognize(self, image: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
        """
        识别车牌
        
        Args:
            image: 原始图像
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            {
                "plate_number": "川A4F68G",
                "plate_type": "蓝牌",
                "confidence": 0.95
            }
        """
        if self.recognizer is None:
            return {
                "plate_number": "",
                "plate_type": "",
                "confidence": 0.0
            }
        
        try:
            # 裁剪车牌区域
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            h, w = image.shape[:2]
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return {"plate_number": "", "plate_type": "", "confidence": 0.0}
            
            # 适当扩大裁剪区域（提高识别率）
            expand_ratio = 0.1
            expand_w = int((x2 - x1) * expand_ratio)
            expand_h = int((y2 - y1) * expand_ratio)
            
            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(w, x2 + expand_w)
            y2 = min(h, y2 + expand_h)
            
            plate_img = image[y1:y2, x1:x2]
            
            if plate_img.size == 0:
                return {"plate_number": "", "plate_type": "", "confidence": 0.0}
            
            # 调用HyperLPR3识别（兼容不同版本）
            results = self._predict(plate_img)
            
            if results and len(results) > 0:
                # 取置信度最高的结果
                best_result = self._extract_best_result(results)
                
                if best_result:
                    return best_result
            
            return {"plate_number": "", "plate_type": "", "confidence": 0.0}
            
        except Exception as e:
            print(f"[车牌识别] 识别失败: {e}")
            import traceback
            traceback.print_exc()
            return {"plate_number": "", "plate_type": "", "confidence": 0.0}
    
    def _predict(self, image: np.ndarray):
        """调用识别器（兼容不同版本和方法名）"""
        try:
            if self._predict_method == '__call__':
                # 直接调用对象
                return self.recognizer(image)
            elif self._predict_method:
                # 调用检测到的方法
                method = getattr(self.recognizer, self._predict_method)
                return method(image)
            else:
                # 最后尝试：直接调用
                if callable(self.recognizer):
                    return self.recognizer(image)
                return []
        except Exception as e:
            print(f"[车牌识别] predict失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_best_result(self, results) -> Dict[str, Any]:
        """提取最佳结果（兼容不同返回格式）"""
        try:
            # 格式1: 列表的字典 [{code, type, confidence}, ...]
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict):
                    best = max(results, key=lambda x: x.get('confidence', 0))
                    
                    plate_number = best.get('code', best.get('plate', ''))
                    plate_type_code = best.get('type', 'unknown')
                    confidence = best.get('confidence', best.get('conf', 0.0))
                    
                    # 映射车牌类型
                    plate_type = PLATE_TYPE_MAP.get(
                        str(plate_type_code).lower(),
                        str(plate_type_code)
                    )
                    
                    return {
                        "plate_number": plate_number,
                        "plate_type": plate_type,
                        "confidence": float(confidence)
                    }
                
                # 格式2: 列表的元组 [(code, confidence, type, bbox), ...]
                # HyperLPR3 常见格式: (车牌号, 置信度, 类型, 边界框)
                elif isinstance(results[0], (tuple, list)):
                    best = max(results, key=lambda x: x[1] if len(x) > 1 else 0)
                    
                    plate_number = best[0] if len(best) > 0 else ''
                    confidence = best[1] if len(best) > 1 else 0.0
                    plate_type_code = best[2] if len(best) > 2 else 'unknown'
                    
                    plate_type = PLATE_TYPE_MAP.get(
                        str(plate_type_code).lower(),
                        str(plate_type_code)
                    )
                    
                    return {
                        "plate_number": plate_number,
                        "plate_type": plate_type,
                        "confidence": float(confidence)
                    }
                
                # 格式3: numpy数组或其他
                elif hasattr(results[0], '__iter__'):
                    # 尝试转换为列表处理
                    first = list(results[0])
                    if len(first) >= 2:
                        return {
                            "plate_number": str(first[0]),
                            "plate_type": PLATE_TYPE_MAP.get(str(first[2]).lower(), "未知") if len(first) > 2 else "未知",
                            "confidence": float(first[1]) if len(first) > 1 else 0.0
                        }
            
            return None
            
        except Exception as e:
            print(f"[车牌识别] 结果解析失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("通用车牌识别器测试")
    print("=" * 70)
    
    # 测试初始化
    recognizer = UniversalPlateRecognizer()
    
    if recognizer.recognizer is None:
        print("\n❌ 识别器初始化失败")
        print("请确保HyperLPR3已正确安装")
    else:
        print(f"\n✅ 识别器初始化成功")
        print(f"   版本: {recognizer.version}")
        print(f"   预测方法: {recognizer._predict_method}")
        
        # 列出识别器的所有方法
        print(f"\n   可用方法:")
        for m in dir(recognizer.recognizer):
            if not m.startswith('_'):
                print(f"     - {m}")
        
        # 创建测试图像
        import numpy as np
        test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        bbox = [10, 10, 190, 90]
        
        print("\n测试识别...")
        result = recognizer.recognize(test_img, bbox)
        print(f"结果: {result}")
    
    print("\n" + "=" * 70)
