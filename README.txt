# NPU gRPC 推理服务

## 文件说明
```
main_grpc_npu.py      - NPU主服务（替换原main_grpc.py）
benchmark_grpc.py     - 压测工具（保持不变）
request_grpc.py       - 测试客户端（保持不变）
```

## 使用方法

### 启动服务
```bash
# 直接启动
python3 main_grpc_npu.py

# 或使用systemd（可选）
python3 main_grpc_npu.py &
```

### 测试
```bash
# 单次测试
python3 request_grpc.py

# 压力测试
python3 benchmark_grpc.py
```

## 关键特性
- ✅ 16 NPU设备轮询负载均衡
- ✅ 24模型LRU缓存
- ✅ 模型路径: /home/yolov5-7.0/modelv1
- ✅ 完全兼容原API

## 配置
修改 `main_grpc_npu.py` 顶部：
```python
MODEL_DIR = "/home/yolov5-7.0/modelv1"  # 模型路径
API_KEY = "api-key"                     # API密钥
NPU_DEVICE_IDS = list(range(16))        # NPU设备
```
