# 无人机图像智能识别 gRPC 接口使用说明

> **版本**：v3.1.0-npu  
> **协议**：gRPC (Protocol Buffers)  
> **更新时间**：2025年  
> **运行环境**：华为昇腾 NPU (Ascend 310P3)

---

## 目录

1. [概述](#1-概述)
2. [接口定义](#2-接口定义)
3. [目标检测接口 Detect](#3-目标检测接口-detect)
4. [变化检测接口 DetectChange](#4-变化检测接口-detectchange)
5. [辅助接口](#5-辅助接口)
6. [算法详细说明](#6-算法详细说明)
7. [客户端调用示例](#7-客户端调用示例)
8. [错误处理](#8-错误处理)
9. [性能优化建议](#9-性能优化建议)
10. [附录](#10-附录)

---

## 1. 概述

本服务基于 gRPC 协议提供无人机航拍图像的智能分析能力，支持 **24 种场景识别算法**。服务端采用华为昇腾 NPU 进行高性能推理，支持 16 设备并行处理。

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| **目标检测** | 支持 23 种场景的目标检测（算法 ID: 1-12, 14-24） |
| **变化检测** | 通过对比两张图片检测场景变化（算法 ID: 13） |
| **车牌识别** | 目标检测基础上增加车牌号码识别（算法 ID: 5） |

### 1.2 服务信息

| 项目 | 值 |
|------|------|
| 服务地址 | `localhost:8000` |
| 协议类型 | gRPC (HTTP/2) |
| 认证方式 | Metadata 携带 `x-api-key` |
| 消息大小限制 | 50MB（发送/接收） |
| 并发线程数 | 32 |

### 1.3 依赖安装

```bash
# Python 客户端依赖
pip install grpcio grpcio-tools protobuf

# 生成 Python gRPC 代码（如需）
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. detection.proto
```

---

## 2. 接口定义

### 2.1 服务定义 (Protobuf)

```protobuf
service DetectionService {
    // 目标检测（算法 1-12, 14-24）
    rpc Detect(DetectRequest) returns (DetectResponse);
    
    // 变化检测（算法 13）
    rpc DetectChange(ChangeDetectRequest) returns (ChangeDetectResponse);
    
    // 健康检查
    rpc HealthCheck(HealthRequest) returns (HealthResponse);
    
    // 版本信息
    rpc GetVersion(VersionRequest) returns (VersionResponse);
}
```

### 2.2 认证方式

所有接口调用需在 gRPC Metadata 中携带 API Key：

```python
metadata = [('x-api-key', 'your-api-key')]
response = stub.Detect(request, metadata=metadata)
```

### 2.3 图片编码规范

| 项目 | 要求 |
|------|------|
| 编码格式 | Base64（**不含** `data:image/...;base64,` 前缀） |
| 图片格式 | JPG / PNG |
| 最大边长 | 4096 像素（超出自动等比缩放） |
| 建议大小 | ≤ 10MB |

**Base64 编码示例**：

```python
import base64

def image_to_base64(path: str) -> str:
    """读取图片文件并转换为 base64（无前缀）"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
```

---

## 3. 目标检测接口 Detect

用于 23 种目标检测类算法（算法 ID: 1-12, 14-24）。

### 3.1 请求消息 DetectRequest

```protobuf
message DetectRequest {
    int32 algorithm_id = 1;      // 算法 ID
    string image = 2;            // 图片 Base64
    float conf_threshold = 3;    // 置信度阈值
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| algorithm_id | int32 | ✅ | 算法 ID（1-12, 14-24，详见第6节） |
| image | string | ✅ | 图片 Base64 编码（无前缀） |
| conf_threshold | float | ❌ | 置信度阈值 [0,1]，默认 0.25 |

### 3.2 响应消息 DetectResponse

```protobuf
message DetectResponse {
    int32 code = 1;              // 状态码
    string message = 2;          // 状态消息
    DetectionData data = 3;      // 检测数据
}

message DetectionData {
    int32 algorithm_id = 1;
    string algorithm_name = 2;
    repeated Detection detections = 3;
    int32 total_count = 4;
    float detect_time = 5;
}

message Detection {
    int32 class_id = 1;
    string class_name = 2;
    string class_name_cn = 3;
    float confidence = 4;
    repeated float bbox = 5;
    // 车牌识别专用字段（仅 algorithm_id=5）
    string plate_number = 6;
    string plate_type = 7;
    float plate_confidence = 8;
}
```

### 3.3 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int32 | 状态码，`200` 表示成功 |
| message | string | 状态描述 |
| data.algorithm_id | int32 | 算法 ID |
| data.algorithm_name | string | 算法名称（中文） |
| data.detections | repeated | 检测结果数组 |
| data.total_count | int32 | 检测目标总数 |
| data.detect_time | float | 推理耗时（秒） |

**Detection 检测结果字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| class_id | int32 | 类别 ID |
| class_name | string | 类别名称（英文） |
| class_name_cn | string | 类别名称（中文） |
| confidence | float | 置信度 [0,1] |
| bbox | repeated float | **归一化**边界框 `[x1, y1, x2, y2]`，范围 0-1 |
| plate_number | string | 车牌号码（仅算法5） |
| plate_type | string | 车牌类型（仅算法5） |
| plate_confidence | float | 车牌识别置信度（仅算法5） |

> **bbox 坐标系**：左上角为原点 (0,0)，坐标已归一化到 [0,1] 范围。  
> 还原像素坐标：`pixel_x = bbox_x * image_width`

### 3.4 成功响应示例

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "algorithm_id": 6,
    "algorithm_name": "交通拥堵识别",
    "detections": [
      {
        "class_id": 0,
        "class_name": "vehicle",
        "class_name_cn": "车辆",
        "confidence": 0.92,
        "bbox": [0.156, 0.234, 0.391, 0.469]
      },
      {
        "class_id": 0,
        "class_name": "vehicle",
        "class_name_cn": "车辆",
        "confidence": 0.87,
        "bbox": [0.5, 0.28, 0.656, 0.484]
      }
    ],
    "total_count": 2,
    "detect_time": 0.045
  }
}
```

### 3.5 车牌识别响应示例（算法 ID=5）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "algorithm_id": 5,
    "algorithm_name": "车牌识别",
    "detections": [
      {
        "class_id": 0,
        "class_name": "license_plate",
        "class_name_cn": "车牌",
        "confidence": 0.95,
        "bbox": [0.35, 0.62, 0.48, 0.71],
        "plate_number": "京A12345",
        "plate_type": "蓝牌",
        "plate_confidence": 0.98
      }
    ],
    "total_count": 1,
    "detect_time": 0.052
  }
}
```

---

## 4. 变化检测接口 DetectChange

用于对比两张图片，检测场景变化区域（算法 ID: 13）。

### 4.1 命令行使用方式

```bash
python request_change_detection.py <img1.png> <img2.png> [output_mask.png]
```

| 参数 | 必填 | 说明 |
|------|------|------|
| img1.png | ✅ | 变化前图片路径 |
| img2.png | ✅ | 变化后图片路径 |
| output_mask.png | ❌ | 输出掩码保存路径（默认: `change_mask.png`） |

**示例**：

```bash
# 基本用法
python request_change_detection.py before.png after.png

# 指定输出路径
python request_change_detection.py before.png after.png result_mask.png
```

### 4.2 请求消息 ChangeDetectRequest

```protobuf
message ChangeDetectRequest {
    string image1 = 1;    // 变化前图片 Base64
    string image2 = 2;    // 变化后图片 Base64
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image1 | string | ✅ | 变化前图片 Base64（无前缀） |
| image2 | string | ✅ | 变化后图片 Base64（无前缀） |

> **注意**：  
>
> - 两张图片应为**同一场景**的不同时间拍摄，输入尺寸应为256×256 
> - 支持 JPG / PNG 格式  
> - 第三个参数（输出路径）是**客户端本地参数**，用于保存服务端返回的 mask 图片，不属于 gRPC 请求字段

### 4.3 响应消息 ChangeDetectResponse

```protobuf
message ChangeDetectResponse {
    int32 code = 1;
    string message = 2;
    string mask = 3;           // 变化掩码 PNG Base64
    int32 width = 4;           // 原图宽度
    int32 height = 5;          // 原图高度
    float change_ratio = 6;    // 变化区域占比
    float detect_time = 7;     // 推理耗时
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int32 | 状态码，`200` 表示成功 |
| message | string | 状态描述 |
| mask | string | 变化掩码图 PNG Base64 编码 |
| width | int32 | 原图宽度（像素） |
| height | int32 | 原图高度（像素） |
| change_ratio | float | 变化区域占比 [0,1] |
| detect_time | float | 推理耗时（秒） |

> **mask 说明**：返回的 mask 是一张 PNG 图片的 Base64 编码。  
> - 黑色像素 (0)：无变化区域  
> - 白色像素 (255)：有变化区域

### 4.4 成功响应示例

```json
{
  "code": 200,
  "message": "success",
  "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
  "width": 1920,
  "height": 1080,
  "change_ratio": 0.0523,
  "detect_time": 0.038
}
```

### 4.5 保存变化掩码

```python
import base64

def save_mask(mask_b64: str, output_path: str):
    """将 base64 编码的掩码保存为图片"""
    img_bytes = base64.b64decode(mask_b64)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"掩码已保存至: {output_path}")
```

---

## 5. 辅助接口

### 5.1 健康检查 HealthCheck

```protobuf
message HealthRequest {}

message HealthResponse {
    string status = 1;
    string device = 2;
    string yolov5_repo = 3;
    int32 models_cached = 4;
    string image_backend = 5;
    string gpu_name = 6;
    float gpu_memory_allocated_mb = 7;
    float gpu_memory_reserved_mb = 8;
}
```

**响应示例**：

```json
{
  "status": "ok",
  "device": "NPU (Ascend 310P3) x16",
  "yolov5_repo": "OM Models",
  "models_cached": 384,
  "image_backend": "opencv",
  "gpu_name": "16 NPU devices",
  "gpu_memory_allocated_mb": 0.0,
  "gpu_memory_reserved_mb": 0.0
}
```

### 5.2 版本信息 GetVersion

```protobuf
message VersionRequest {}

message VersionResponse {
    string version = 1;
    string mode = 2;
    string pytorch_version = 3;
    string opencv_version = 4;
    string device = 5;
    bool cuda_available = 6;
    float default_conf_threshold = 7;
    repeated int32 algo_supported = 8;
}
```

**响应示例**：

```json
{
  "version": "3.1.0-npu",
  "mode": "offline-npu-preload",
  "pytorch_version": "N/A (OM Runtime)",
  "opencv_version": "4.8.1",
  "device": "Ascend NPU x16",
  "cuda_available": false,
  "default_conf_threshold": 0.25,
  "algo_supported": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
}
```

---

## 6. 算法详细说明

### 6.1 算法总览

| ID | 算法名称 | 接口 | 典型应用场景 |
|----|---------|------|-------------|
| 1 | 松线虫害识别 | Detect | 林业病虫害监测 |
| 2 | 河道淤积识别 | Detect | 水利设施巡检 |
| 3 | 漂浮物识别 | Detect | 水域环境监测 |
| 4 | 游泳涉水识别 | Detect | 水域安全管控 |
| 5 | 车牌识别 | Detect | 交通管理（含号码识别） |
| 6 | 交通拥堵识别 | Detect | 交通监控 |
| 7 | 路面破损识别 | Detect | 道路养护 |
| 8 | 路面污染 | Detect | 环境卫生监管 |
| 9 | 人群聚集识别 | Detect | 公共安全管理 |
| 10 | 非法垂钓识别 | Detect | 水域管理 |
| 11 | 施工识别 | Detect | 工地监管 |
| 12 | 秸秆焚烧 | Detect | 环保执法 |
| **13** | **变化检测** | **DetectChange** | **场景变化分析** |
| 14 | 占道经营识别 | Detect | 城市管理 |
| 15 | 垃圾堆放识别 | Detect | 环境卫生监管 |
| 16 | 裸土未覆盖识别 | Detect | 扬尘治理 |
| 17 | 建控区违建识别 | Detect | 城市规划执法 |
| 18 | 烟火识别 | Detect | 消防安全 |
| 19 | 光伏板缺陷检测 | Detect | 设备运维 |
| 20 | 园区夜间入侵检测 | Detect | 园区安防 |
| 21 | 园区外立面病害识别 | Detect | 建筑安全 |
| 22 | 罂粟识别 | Detect | 禁毒执法 |
| 23 | 作物倒伏检测 | Detect | 农业灾害评估 |
| 24 | 林业侵占 | Detect | 林地保护 |

### 6.2 各算法检测类别详情

#### 算法 1：松线虫害识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | dead | 死亡 |
| 1 | heavy | 重度患病 |
| 2 | light | 轻度患病 |

#### 算法 2：河道淤积识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | waterpollution | 水污染 |
| 1 | floatingdebris | 漂浮碎片 |
| 2 | abandonedships | 废弃船只 |
| 3 | fishingandaquaculture | 渔业和水产养殖 |
| 4 | waste | 垃圾 |

#### 算法 3：漂浮物识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | bottle | 瓶子 |
| 1 | grass | 草 |
| 2 | branch | 树枝 |
| 3 | milk-box | 牛奶盒 |
| 4 | plastic-bag | 塑料袋 |
| 5 | plastic-garbage | 塑料垃圾袋 |
| 6 | ball | 球 |
| 7 | leaf | 叶子 |

#### 算法 4：游泳涉水识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | ignored | 忽略 |
| 1 | swimmer | 游泳者 |
| 2 | boat | 船 |
| 3 | jetski | 水上摩托艇 |
| 4 | life_saving_appliances | 救生设备 |
| 5 | buoy | 浮标 |

#### 算法 5：车牌识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | license_plate | 车牌 |

> 额外返回字段：`plate_number`（车牌号）、`plate_type`（车牌类型）、`plate_confidence`

#### 算法 6：交通拥堵识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | vehicle | 车辆 |

#### 算法 7：路面破损识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | Alligator Crack | 龟裂 |
| 1 | Longitudinal Crack | 纵向裂缝 |
| 2 | Longitudinal Patch | 纵向修补块 |
| 3 | Manhole Cover | 检查井井盖 |
| 4 | Pothole | 坑洞 |
| 5 | Transverse Crack | 横向裂缝 |
| 6 | Transverse Patch | 横向修补块 |

#### 算法 8：路面污染

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | BAD_BILLBOARD | 违规广告牌 |
| 1 | BROKEN_SIGNAGE | 破损标识牌 |
| 2 | CLUTTER_SIDEWALK | 人行道杂物堆积 |
| 3 | CONSTRUCTION_ROAD | 施工路段 |
| 4 | FADED_SIGNAGE | 褪色标识牌 |
| 5 | GARBAGE | 垃圾堆积 |
| 6 | GRAFFITI | 涂鸦乱画 |
| 7 | POTHOLES | 路面坑洞 |
| 8 | SAND_ON_ROAD | 路面积沙 |
| 9 | UNKEPT_FACADE | 建筑外立面破损 |

#### 算法 9：人群聚集识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | car | 车 |
| 1 | people | 人 |

#### 算法 10：非法垂钓识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | ShuiBianDiaoYu | 水边钓鱼 |
| 1 | YouYongNiShui | 游泳溺水 |
| 2 | DiaoYuSan | 钓鱼伞 |
| 3 | boat | 船 |

#### 算法 11：施工识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | crane | 起重机 |
| 1 | excavator | 挖掘机 |
| 2 | tractor | 拖拉机 |
| 3 | truck | 卡车 |

#### 算法 12：秸秆焚烧

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | straw | 秸秆堆 |

#### 算法 14：占道经营识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | zdjy | 占道经营 |

#### 算法 15：垃圾堆放识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | Bench | 长椅 |
| 1 | Commercial_Trash | 商业垃圾 |
| 2 | Dumping-sites | 非法倾倒点 |
| 3 | Green_Land | 绿地 |
| 4 | Hole | 孔洞 |
| 5 | Jersey_Barrier | 泽西护栏 |
| 6 | Land | 地块 |
| 7 | Raw_Material | 原材料 |
| 8 | Trash | 生活垃圾 |

#### 算法 16：裸土未覆盖识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | trash | 垃圾 |
| 1 | bare_soil | 裸土 |

#### 算法 17：建控区违建识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | blue_canopy | 蓝色天篷 |
| 1 | others | 其他违建 |
| 2 | green_shack | 改装绿色小屋 |

#### 算法 18：烟火识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | smoke | 烟雾 |
| 1 | fire | 火 |

#### 算法 19：光伏板缺陷检测

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | defected-pv-cells | 有缺陷的光伏电池 |

#### 算法 20：园区夜间入侵检测

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | person | 人 |
| 1 | car | 车 |
| 2 | bicycle | 自行车 |

#### 算法 21：园区外立面病害识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | wall_corrosion | 墙体腐蚀 |
| 1 | wall_crack | 墙体开裂 |
| 2 | wall_deterioration | 墙体劣化 |
| 3 | wall_mold | 墙模 |
| 4 | wall_stain | 墙面污渍 |

#### 算法 22：罂粟识别

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | poppy-opium | 罂粟 |

#### 算法 23：作物倒伏检测

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | Lodged | 作物倒伏 |

#### 算法 24：林业侵占

| class_id | class_name | class_name_cn |
|----------|------------|---------------|
| 0 | backhoe_loader | 反铲装载机 |
| 1 | compactor | 压路机 |
| 2 | concrete_mixer_truck | 混凝土搅拌车 |
| 3 | dozer | 推土机 |
| 4 | dump_truck | 倾卸卡车 |
| 5 | excavator | 挖掘机 |
| 6 | grader | 评分员 |
| 7 | helmet | 安全头盔 |
| 8 | mobile_crane | 移动式起重机 |
| 9 | person | 人 |
| 10 | tower_crane | 塔式起重机 |
| 11 | vest | 背心 |
| 12 | wheel_loader | 轮式装载机 |

---

## 7.服务端启动与管理

### 7.1 启动环境配置

```
conda activate yolov5
```

### 7.2 切换到目标目录

```
cd /home/yolov5-7.0-2.0/modelv1/
```

### 7.3 添加可执行权限（如果没有）

```
chmod +x service.sh
```

### 7.4 启动服务

```
./service.sh start
```

### 7.5 停止服务

```
./service.sh stop
```

### 7.6 重启服务

```
./service.sh restart
```

### 7.7 查看服务状态

```
./service.sh status
```

### 7.8 查看实时日志（Ctrl+C 退出）

```
./service.sh log
```

### 7.9 查看最近100行日志

```
./service.sh tail
```

### 7.10 查看最近50行日志

```
./service.sh tail 50
```

### 7.11 清理日志件

```
./service.sh clean
```

### 7.12 帮助

```
./service.sh help
```



````
## 效果示例

**启动服务：**
```
========================================
  启动 uav-detection 服务
========================================
[INFO] 文件描述符限制: 65535
[INFO] Python 路径: /root/miniconda3/envs/yolov5/bin/python
[INFO] 启动服务...
[INFO] 等待服务启动...
[INFO] 服务启动成功！

  PID:      12345
  端口:     8000
  日志:     /home/yolov5-7.0-2.0/modelv1/logs/uav-detection.log
  工作目录: /home/yolov5-7.0-2.0/modelv1

  查看日志: ./service.sh log
```

**查看状态：**
```
========================================
  uav-detection 服务状态
========================================

  状态:     运行中
  PID:      12345
  CPU:      2.5%
  内存:     15.3%
  运行时间: 01:23:45
  端口:     8000 (监听中)
  日志:     /home/.../logs/uav-detection.log

最近日志:
----------------------------------------
✅ 服务就绪
   可用模型: 24 个
🚀 gRPC 服务已启动，监听端口 8000
----------------------------------------
````

## 8. 客户端调用示例

### 8.1 Python - 目标检测

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目标检测 gRPC 客户端示例"""

import base64
import json
import grpc
import detection_pb2
import detection_pb2_grpc

# 配置
GRPC_SERVER = "localhost:8000"
API_KEY = "api-key"

def image_to_base64(path: str) -> str:
    """读取图片并转换为 base64（无前缀）"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def detection_to_dict(det):
    """将检测结果转为字典"""
    result = {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "class_name_cn": det.class_name_cn,
        "confidence": round(det.confidence, 4),
        "bbox": [round(x, 6) for x in det.bbox]
    }
    # 车牌识别额外字段
    if det.plate_number:
        result["plate_number"] = det.plate_number
        result["plate_type"] = det.plate_type
        result["plate_confidence"] = round(det.plate_confidence, 4)
    return result

def detect(image_path: str, algorithm_id: int, conf_threshold: float = 0.25):
    """执行目标检测"""
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        
        # 构建请求
        request = detection_pb2.DetectRequest(
            algorithm_id=algorithm_id,
            image=image_to_base64(image_path),
            conf_threshold=conf_threshold
        )
        
        # 添加认证
        metadata = [('x-api-key', API_KEY)]
        
        try:
            # 发送请求
            response = stub.Detect(request, metadata=metadata)
            
            # 处理响应
            result = {
                "code": response.code,
                "message": response.message,
                "data": {
                    "algorithm_id": response.data.algorithm_id,
                    "algorithm_name": response.data.algorithm_name,
                    "detections": [detection_to_dict(d) for d in response.data.detections],
                    "total_count": response.data.total_count,
                    "detect_time": round(response.data.detect_time, 3)
                }
            }
            
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result
            
        except grpc.RpcError as e:
            print(f"gRPC 错误: {e.code()} - {e.details()}")
            return None

if __name__ == "__main__":
    # 示例：交通拥堵识别
    detect("traffic.jpg", algorithm_id=6, conf_threshold=0.3)
```

### 8.2 Python - 变化检测

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""变化检测 gRPC 客户端示例"""

import base64
import grpc
import detection_pb2
import detection_pb2_grpc

GRPC_SERVER = "localhost:8000"
API_KEY = "api-key"

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_mask(b64_str: str, output_path: str):
    """保存 base64 掩码为图片"""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_str))

def detect_change(img1_path: str, img2_path: str, output_mask: str = "change_mask.png"):
    """执行变化检测"""
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        
        request = detection_pb2.ChangeDetectRequest(
            image1=image_to_base64(img1_path),
            image2=image_to_base64(img2_path)
        )
        
        metadata = [('x-api-key', API_KEY)]
        
        try:
            response = stub.DetectChange(request, metadata=metadata)
            
            print(f"状态: {response.code} - {response.message}")
            print(f"图片尺寸: {response.width} x {response.height}")
            print(f"变化占比: {response.change_ratio * 100:.2f}%")
            print(f"推理耗时: {response.detect_time * 1000:.2f} ms")
            
            # 保存掩码
            save_mask(response.mask, output_mask)
            print(f"掩码已保存: {output_mask}")
            
            return response
            
        except grpc.RpcError as e:
            print(f"gRPC 错误: {e.code()} - {e.details()}")
            return None

if __name__ == "__main__":
    detect_change("before.png", "after.png", "change_mask.png")
```

### 8.3 Python - 健康检查与版本信息

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""辅助接口调用示例"""

import json
import grpc
import detection_pb2
import detection_pb2_grpc

GRPC_SERVER = "localhost:8000"

def health_check():
    """健康检查"""
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        response = stub.HealthCheck(detection_pb2.HealthRequest())
        
        print("=== 健康检查 ===")
        print(json.dumps({
            "status": response.status,
            "device": response.device,
            "models_cached": response.models_cached,
            "image_backend": response.image_backend
        }, indent=2, ensure_ascii=False))

def get_version():
    """获取版本信息"""
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        response = stub.GetVersion(detection_pb2.VersionRequest())
        
        print("=== 版本信息 ===")
        print(json.dumps({
            "version": response.version,
            "mode": response.mode,
            "device": response.device,
            "opencv_version": response.opencv_version,
            "default_conf_threshold": response.default_conf_threshold,
            "algo_supported": list(response.algo_supported)
        }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    health_check()
    get_version()
```

### 8.4 坐标还原示例

```python
def restore_pixel_coords(bbox: list, img_width: int, img_height: int) -> dict:
    """
    将归一化坐标还原为像素坐标
    
    Args:
        bbox: 归一化边界框 [x1, y1, x2, y2]，范围 0-1
        img_width: 原图宽度
        img_height: 原图高度
    
    Returns:
        像素坐标字典
    """
    x1, y1, x2, y2 = bbox
    return {
        "x1": int(x1 * img_width),
        "y1": int(y1 * img_height),
        "x2": int(x2 * img_width),
        "y2": int(y2 * img_height),
        "width": int((x2 - x1) * img_width),
        "height": int((y2 - y1) * img_height)
    }

# 使用示例
bbox_norm = [0.156, 0.234, 0.391, 0.469]
pixel_coords = restore_pixel_coords(bbox_norm, img_width=1920, img_height=1080)
print(pixel_coords)
# 输出: {'x1': 299, 'y1': 252, 'x2': 750, 'y2': 506, 'width': 451, 'height': 254}
```

---

## 9. 错误处理

### 9.1 gRPC 状态码

| StatusCode | 说明 | 处理建议 |
|------------|------|---------|
| OK | 成功 | — |
| UNAUTHENTICATED | 认证失败 | 检查 `x-api-key` 是否正确 |
| NOT_FOUND | 算法不存在 | 确认 `algorithm_id` 在有效范围内 |
| INVALID_ARGUMENT | 参数错误 | 检查 Base64 编码、必填字段 |
| INTERNAL | 服务器内部错误 | 查看服务端日志 |
| UNAVAILABLE | 服务不可用 | 确认服务已启动，网络可达 |

### 9.2 错误处理示例

```python
import grpc

try:
    response = stub.Detect(request, metadata=metadata)
except grpc.RpcError as e:
    status_code = e.code()
    details = e.details()
    
    if status_code == grpc.StatusCode.UNAUTHENTICATED:
        print("认证失败，请检查 API Key")
    elif status_code == grpc.StatusCode.NOT_FOUND:
        print(f"算法不存在: {details}")
    elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
        print(f"参数错误: {details}")
    elif status_code == grpc.StatusCode.UNAVAILABLE:
        print("服务不可用，请确认服务已启动")
    else:
        print(f"未知错误 [{status_code}]: {details}")
```

### 9.3 常见错误及解决方案

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `未授权: API Key 无效` | API Key 不匹配 | 检查 metadata 中的 `x-api-key` |
| `需要提供 image 字段` | 缺少图片数据 | 确保 `image` 字段已填充 |
| `base64 解码失败` | Base64 格式错误 | 确保无 `data:image/...` 前缀 |
| `图像解码失败` | 图片格式不支持 | 使用 JPG/PNG 格式 |
| `模型不存在` | algorithm_id 无对应模型 | 使用 `GetVersion` 查询支持的算法 |
| `failed to connect` | 网络不通 | 检查服务地址和端口 |

---

## 10. 性能优化建议

### 10.1 图片优化

| 优化项 | 建议 | 说明 |
|-------|------|------|
| 分辨率 | 适当降低 | 超过 4096 会被自动缩放 |
| 压缩质量 | JPEG 80-90% | 平衡质量与传输效率 |
| 文件大小 | ≤ 10MB | 减少网络传输时间 |

### 10.2 批量处理

```python
import concurrent.futures

def batch_detect(image_paths: list, algorithm_id: int, max_workers: int = 4):
    """批量检测"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(detect, path, algorithm_id): path 
            for path in image_paths
        }
        
        results = {}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as e:
                results[path] = {"error": str(e)}
        
        return results
```

### 10.3 连接复用

```python
# 推荐：复用 channel 和 stub
channel = grpc.insecure_channel(GRPC_SERVER)
stub = detection_pb2_grpc.DetectionServiceStub(channel)

# 多次调用使用同一个 stub
for image_path in image_list:
    response = stub.Detect(request, metadata=metadata)

# 使用完毕后关闭
channel.close()
```

---

## 11. 附录

### 11.1 Protobuf 完整定义

```protobuf
syntax = "proto3";
package detection;

service DetectionService {
    rpc Detect(DetectRequest) returns (DetectResponse);
    rpc DetectChange(ChangeDetectRequest) returns (ChangeDetectResponse);
    rpc HealthCheck(HealthRequest) returns (HealthResponse);
    rpc GetVersion(VersionRequest) returns (VersionResponse);
}

message DetectRequest {
    int32 algorithm_id = 1;
    string image = 2;
    float conf_threshold = 3;
}

message DetectResponse {
    int32 code = 1;
    string message = 2;
    DetectionData data = 3;
}

message DetectionData {
    int32 algorithm_id = 1;
    string algorithm_name = 2;
    repeated Detection detections = 3;
    int32 total_count = 4;
    float detect_time = 5;
}

message Detection {
    int32 class_id = 1;
    string class_name = 2;
    string class_name_cn = 3;
    float confidence = 4;
    repeated float bbox = 5;
    string plate_number = 6;
    string plate_type = 7;
    float plate_confidence = 8;
}

message ChangeDetectRequest {
    string image1 = 1;
    string image2 = 2;
}

message ChangeDetectResponse {
    int32 code = 1;
    string message = 2;
    string mask = 3;
    int32 width = 4;
    int32 height = 5;
    float change_ratio = 6;
    float detect_time = 7;
}

message HealthRequest {}

message HealthResponse {
    string status = 1;
    string device = 2;
    string yolov5_repo = 3;
    int32 models_cached = 4;
    string image_backend = 5;
    string gpu_name = 6;
    float gpu_memory_allocated_mb = 7;
    float gpu_memory_reserved_mb = 8;
}

message VersionRequest {}

message VersionResponse {
    string version = 1;
    string mode = 2;
    string pytorch_version = 3;
    string opencv_version = 4;
    string device = 5;
    bool cuda_available = 6;
    float default_conf_threshold = 7;
    repeated int32 algo_supported = 8;
}
```

### 11.2 服务端处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        客户端请求                                │
│  DetectRequest { algorithm_id, image(base64), conf_threshold }  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 认证校验 (x-api-key)                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Base64 解码 → 原图 (W × H)                                   │
│     - 自动缩放超大图片 (>4096px)                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 图像预处理                                                   │
│     - 等比缩放 + letterbox padding → 640×640                     │
│     - 归一化、通道转换                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. NPU 推理 (昇腾 310P3)                                        │
│     - 设备轮询负载均衡                                           │
│     - OM 模型推理                                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 后处理                                                       │
│     - NMS 去重                                                   │
│     - 坐标映射回原图                                             │
│     - 归一化坐标 (0-1)                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        响应返回                                  │
│  DetectResponse { code, message, data { detections, ... } }     │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3 性能指标参考

| 指标 | 数值 |
|------|------|
| 单次推理延迟 | 10-50 ms |
| 并发线程数 | 32 |
| NPU 设备数 | 16 |
| 预加载模型数 | 24 × 16 = 384 实例 |
| 消息大小限制 | 50 MB |
| 服务启动时间 | ~120 秒（含模型预加载） |

---

*文档版本: v3.1.0-npu | gRPC 接口使用说明*
