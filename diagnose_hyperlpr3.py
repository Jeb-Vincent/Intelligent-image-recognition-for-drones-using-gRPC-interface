#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperLPR3 导入诊断脚本
"""

print("=" * 70)
print("HyperLPR3 导入诊断")
print("=" * 70)

# 测试1: 检查hyperlpr3包
print("\n[测试1] 检查hyperlpr3包...")
try:
    import hyperlpr3
    print(f"  ✓ hyperlpr3包已安装")
    print(f"  版本: {hyperlpr3.__version__ if hasattr(hyperlpr3, '__version__') else '未知'}")
    print(f"  路径: {hyperlpr3.__file__}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    exit(1)

# 测试2: 查看可用的类和函数
print("\n[测试2] 查看hyperlpr3中的可用对象...")
import hyperlpr3
available = [name for name in dir(hyperlpr3) if not name.startswith('_')]
print(f"  可用对象: {available}")

# 测试3: 尝试不同的导入方式
print("\n[测试3] 尝试不同的导入方式...")

# 方式1: LicensePlateCN (最常见)
try:
    from hyperlpr3 import LicensePlateCN
    print("  ✓ 方式1成功: from hyperlpr3 import LicensePlateCN")
    METHOD = 1
except ImportError as e:
    print(f"  ✗ 方式1失败: {e}")
    METHOD = 0

# 方式2: LicensePlateCatcher (旧版本)
if METHOD == 0:
    try:
        from hyperlpr3 import LicensePlateCatcher
        print("  ✓ 方式2成功: from hyperlpr3 import LicensePlateCatcher")
        METHOD = 2
    except ImportError as e:
        print(f"  ✗ 方式2失败: {e}")

# 方式3: HyperLPR3 类
if METHOD == 0:
    try:
        from hyperlpr3 import HyperLPR3
        print("  ✓ 方式3成功: from hyperlpr3 import HyperLPR3")
        METHOD = 3
    except ImportError as e:
        print(f"  ✗ 方式3失败: {e}")

# 方式4: 直接使用模块
if METHOD == 0:
    try:
        import hyperlpr3 as lpr3
        if hasattr(lpr3, 'predict'):
            print("  ✓ 方式4成功: import hyperlpr3 (直接使用)")
            METHOD = 4
        else:
            print("  ✗ 方式4失败: 没有predict方法")
    except Exception as e:
        print(f"  ✗ 方式4失败: {e}")

# 测试4: 尝试初始化
if METHOD > 0:
    print(f"\n[测试4] 尝试初始化 (使用方式{METHOD})...")
    try:
        if METHOD == 1:
            from hyperlpr3 import LicensePlateCN
            recognizer = LicensePlateCN()
            print("  ✓ 初始化成功")
        elif METHOD == 2:
            from hyperlpr3 import LicensePlateCatcher
            recognizer = LicensePlateCatcher()
            print("  ✓ 初始化成功")
        elif METHOD == 3:
            from hyperlpr3 import HyperLPR3
            recognizer = HyperLPR3()
            print("  ✓ 初始化成功")
        elif METHOD == 4:
            import hyperlpr3 as lpr3
            # 直接使用模块级别的函数
            print("  ✓ 使用模块级别函数")
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

# 测试5: 测试识别功能
if METHOD > 0:
    print(f"\n[测试5] 测试识别功能...")
    try:
        import numpy as np
        import cv2
        
        # 创建测试图像
        test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        if METHOD == 1:
            from hyperlpr3 import LicensePlateCN
            recognizer = LicensePlateCN()
            results = recognizer.predict(test_img)
            print(f"  ✓ predict方法可用")
            print(f"  结果类型: {type(results)}")
            
        elif METHOD == 2:
            from hyperlpr3 import LicensePlateCatcher
            recognizer = LicensePlateCatcher()
            results = recognizer.predict(test_img)
            print(f"  ✓ predict方法可用")
            print(f"  结果类型: {type(results)}")
            
        elif METHOD == 3:
            from hyperlpr3 import HyperLPR3
            recognizer = HyperLPR3()
            results = recognizer.predict(test_img)
            print(f"  ✓ predict方法可用")
            print(f"  结果类型: {type(results)}")
            
        elif METHOD == 4:
            import hyperlpr3 as lpr3
            results = lpr3.predict(test_img)
            print(f"  ✓ 模块级predict方法可用")
            print(f"  结果类型: {type(results)}")
            
    except Exception as e:
        print(f"  ✗ 识别测试失败: {e}")
        import traceback
        traceback.print_exc()

# 总结
print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)

if METHOD == 0:
    print("\n❌ 未找到可用的导入方式")
    print("\n建议:")
    print("  1. 重新安装 HyperLPR3:")
    print("     pip uninstall hyperlpr3 -y")
    print("     pip install hyperlpr3 --break-system-packages")
    print("  2. 或尝试安装特定版本:")
    print("     pip install hyperlpr3==0.0.1 --break-system-packages")
else:
    print(f"\n✅ 找到可用的导入方式: 方式{METHOD}")
    print("\n推荐的代码:")
    
    if METHOD == 1:
        print("""
from hyperlpr3 import LicensePlateCN

# 初始化
recognizer = LicensePlateCN()

# 使用
results = recognizer.predict(image)
""")
    elif METHOD == 2:
        print("""
from hyperlpr3 import LicensePlateCatcher

# 初始化
recognizer = LicensePlateCatcher()

# 使用
results = recognizer.predict(image)
""")
    elif METHOD == 3:
        print("""
from hyperlpr3 import HyperLPR3

# 初始化
recognizer = HyperLPR3()

# 使用
results = recognizer.predict(image)
""")
    elif METHOD == 4:
        print("""
import hyperlpr3 as lpr3

# 直接使用
results = lpr3.predict(image)
""")

print("\n" + "=" * 70)
