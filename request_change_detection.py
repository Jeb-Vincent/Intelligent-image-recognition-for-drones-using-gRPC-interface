#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化检测 gRPC 客户端示例
算法13：输入输出都是 base64 编码的 PNG 图片

使用方法:
    python request_change_detection.py <img1.png> <img2.png> [output_mask.png]
"""

import base64
import sys
import os
import grpc
import detection_pb2
import detection_pb2_grpc

# 配置
GRPC_SERVER = "localhost:8000"
API_KEY = "api-key"


def image_to_base64(path: str) -> str:
    """读取图片文件并转换为 base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_image(b64_str: str, output_path: str):
    """将 base64 字符串保存为图片文件"""
    img_bytes = base64.b64decode(b64_str)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"[保存] Mask 已保存至: {output_path}")


def detect_change(img1_path: str, img2_path: str, output_mask_path: str = "change_mask.png"):
    """
    执行变化检测
    
    Args:
        img1_path: 第一张PNG图片路径
        img2_path: 第二张PNG图片路径
        output_mask_path: 输出mask PNG图片路径
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        
        print("=" * 50)
        print("变化检测请求 (Base64 PNG)")
        print("=" * 50)
        print(f"图片1: {img1_path}")
        print(f"图片2: {img2_path}")
        
        # 转换为 base64
        img1_b64 = image_to_base64(img1_path)
        img2_b64 = image_to_base64(img2_path)
        
        print(f"图片1 base64长度: {len(img1_b64)} 字符")
        print(f"图片2 base64长度: {len(img2_b64)} 字符")
        
        # 创建请求
        request = detection_pb2.ChangeDetectRequest(
            image1=img1_b64,
            image2=img2_b64
        )
        
        metadata = [('x-api-key', API_KEY)]
        
        try:
            print("\n正在执行推理...")
            
            # 调用 DetectChange 接口
            response = stub.DetectChange(request, metadata=metadata)
            
            print("\n" + "=" * 50)
            print("检测结果")
            print("=" * 50)
            print(f"状态码: {response.code}")
            print(f"消息: {response.message}")
            print(f"图片尺寸: {response.width} x {response.height}")
            print(f"变化区域占比: {response.change_ratio * 100:.2f}%")
            print(f"推理耗时: {response.detect_time * 1000:.2f} ms")
            print(f"Mask base64长度: {len(response.mask)} 字符")
            
            # 保存 mask
            base64_to_image(response.mask, output_mask_path)
            
            return {
                "code": response.code,
                "message": response.message,
                "width": response.width,
                "height": response.height,
                "change_ratio": response.change_ratio,
                "detect_time": response.detect_time,
                "mask_path": output_mask_path
            }
            
        except grpc.RpcError as e:
            print(f"\ngRPC 错误:")
            print(f"  状态码: {e.code()}")
            print(f"  详情: {e.details()}")
            return None


def main():
    if len(sys.argv) < 3:
        print("=" * 60)
        print("变化检测 gRPC 客户端 (Base64 PNG)")
        print("=" * 60)
        print("\n使用方法:")
        print("  python request_change_detection.py <img1.png> <img2.png> [output.png]")
        print("\n示例:")
        print("  python request_change_detection.py before.png after.png mask.png")
        print("\n输入输出格式:")
        print("  输入: PNG 图片 -> base64 编码")
        print("  输出: base64 编码 -> PNG mask 图片")
        print("        黑色(0)=无变化, 白色(255)=有变化")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "change_mask.png"
    
    if not os.path.exists(img1_path):
        print(f"错误: 找不到文件 {img1_path}")
        sys.exit(1)
    if not os.path.exists(img2_path):
        print(f"错误: 找不到文件 {img2_path}")
        sys.exit(1)
    
    result = detect_change(img1_path, img2_path, output_path)
    
    if result:
        print("\n" + "=" * 50)
        print("✅ 完成!")
        print("=" * 50)


if __name__ == "__main__":
    main()
