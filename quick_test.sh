#!/bin/bash

cd /home/yolov5-7.0/modelv1

echo "=========================================="
echo "准备测试环境"
echo "=========================================="

# 1. 创建推理脚本（已在上面）
if [ ! -f "infer_om_yolo.py" ]; then
    echo "⚠️  请先创建 infer_om_yolo.py"
    exit 1
fi

# 2. 查找测试图片
echo -e "\n=== 查找测试图片 ==="
TEST_IMG=""

# 优先使用 YOLOv5 自带图片
if [ -f "/home/yolov5-7.0/data/images/bus.jpg" ]; then
    TEST_IMG="/home/yolov5-7.0/data/images/bus.jpg"
    echo "✅ 使用 YOLOv5 自带图片: bus.jpg"
elif [ -f "/home/yolov5-7.0/data/images/zidane.jpg" ]; then
    TEST_IMG="/home/yolov5-7.0/data/images/zidane.jpg"
    echo "✅ 使用 YOLOv5 自带图片: zidane.jpg"
else
    # 查找任意 jpg
    TEST_IMG=$(find /home -name "*.jpg" 2>/dev/null | head -1)
    if [ -n "$TEST_IMG" ]; then
        echo "✅ 找到测试图片: $TEST_IMG"
    else
        echo "❌ 未找到测试图片"
        echo "   请手动指定图片路径"
        exit 1
    fi
fi

# 3. 检查 OM 模型
echo -e "\n=== 检查 OM 模型 ==="
OM_MODEL=""
if [ -f "人脸检测_bs1.om" ]; then
    OM_MODEL="人脸检测_bs1.om"
    echo "✅ 使用: 人脸检测_bs1.om"
else
    # 查找任意 OM 模型
    OM_MODEL=$(ls *_bs1.om 2>/dev/null | head -1)
    if [ -n "$OM_MODEL" ]; then
        echo "✅ 使用: $OM_MODEL"
    else
        echo "❌ 未找到 OM 模型"
        echo "   请先转换模型"
        exit 1
    fi
fi

# 4. 运行推理
echo -e "\n=== 运行推理测试 ==="
python3 infer_om_yolo.py "$OM_MODEL" "$TEST_IMG"

echo -e "\n=========================================="
echo "测试完成"
echo "=========================================="
