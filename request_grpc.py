#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detection API - gRPC Client
gRPC client example code
"""

import base64
import json
import grpc
import detection_pb2
import detection_pb2_grpc

# Configuration
GRPC_SERVER = "localhost:8000"
API_KEY = "api-key"

def to_base64_no_prefix(path: str) -> str:
    """Convert image to base64 encoding (without data: prefix)"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def detection_to_dict(detection):
    """Convert protobuf Detection to dict"""
    result = {
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "class_name_cn": detection.class_name_cn,
        "confidence": detection.confidence,
        "bbox": list(detection.bbox)
    }
    
    # 仅当有车牌信息时添加
    if detection.plate_number:
        result["plate_number"] = detection.plate_number
        result["plate_type"] = detection.plate_type
        result["plate_confidence"] = detection.plate_confidence
    
    return result

def response_to_dict(response):
    """Convert protobuf DetectResponse to dict (same format as REST API)"""
    return {
        "code": response.code,
        "message": response.message,
        "data": {
            "algorithm_id": response.data.algorithm_id,
            "algorithm_name": response.data.algorithm_name,
            "detections": [detection_to_dict(det) for det in response.data.detections],
            "total_count": response.data.total_count,
            "detect_time": response.data.detect_time
        }
    }

def main():
    # Create gRPC channel
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        
        # Prepare request
        request = detection_pb2.DetectRequest(
            algorithm_id=10,
            image=to_base64_no_prefix("test10.jpg"),
            conf_threshold=0.25
        )
        
        # Add metadata (including API Key)
        metadata = [('x-api-key', API_KEY)]
        
        try:
            # Send request
            print("Sending detection request...")
            response = stub.Detect(request, metadata=metadata)
            
            # Convert to dict format (same as REST API)
            result_dict = response_to_dict(response)
            
            print(f"\nStatus code: {response.code}")
            print(f"Message: {response.message}")
            print("\nOriginal response object:")
            print(response)
            print("\nBeautified JSON (same format as REST API):")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
            
        except grpc.RpcError as e:
            print(f"gRPC Error:")
            print(f"  Status code: {e.code()}")
            print(f"  Details: {e.details()}")

def test_health_check():
    """Test health check"""
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        request = detection_pb2.HealthRequest()
        
        try:
            response = stub.HealthCheck(request)
            print("\nHealth check result:")
            print(json.dumps({
                "status": response.status,
                "device": response.device,
                "yolov5_repo": response.yolov5_repo,
                "models_cached": response.models_cached,
                "image_backend": response.image_backend,
                "gpu_name": response.gpu_name or "N/A",
                "gpu_memory_allocated_mb": response.gpu_memory_allocated_mb,
                "gpu_memory_reserved_mb": response.gpu_memory_reserved_mb
            }, indent=2, ensure_ascii=False))
        except grpc.RpcError as e:
            print(f"Health check failed: {e.details()}")

def test_version():
    """Test version info"""
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        request = detection_pb2.VersionRequest()
        
        try:
            response = stub.GetVersion(request)
            print("\nVersion info:")
            print(json.dumps({
                "version": response.version,
                "mode": response.mode,
                "pytorch_version": response.pytorch_version,
                "opencv_version": response.opencv_version,
                "device": response.device,
                "cuda_available": response.cuda_available,
                "default_conf_threshold": response.default_conf_threshold,
                "algo_supported": list(response.algo_supported)
            }, indent=2, ensure_ascii=False))
        except grpc.RpcError as e:
            print(f"Get version info failed: {e.details()}")

if __name__ == "__main__":
    # Main detection request
    main()
    
    # Optional: test other APIs
    # test_health_check()
    # test_version()