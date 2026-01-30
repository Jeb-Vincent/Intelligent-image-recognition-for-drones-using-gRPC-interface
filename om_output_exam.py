#!/usr/bin/env python3
"""验证 OM 模型输出"""
import acl

def check_om_model(model_path: str, device_id: int = 0):
    acl.init()
    acl.rt.set_device(device_id)
    context, _ = acl.rt.create_context(device_id)
    
    model_id, ret = acl.mdl.load_from_file(model_path)
    if ret != 0:
        print(f"加载模型失败: {ret}")
        return
    
    model_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(model_desc, model_id)
    
    input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
    
    # FP16 输出，每个元素 2 字节
    num_elements = output_size // 2
    
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"输入字节数: {input_size}")
    print(f"输出字节数: {output_size}")
    print(f"输出元素数: {num_elements}")
    print("=" * 50)
    
    # 分析可能的形状
    if num_elements == 42000:  # 5 * 8400
        print("正确！YOLOv11 格式 [1, 5, 8400]")
        print(f"   predictions=8400, features=5")
    elif num_elements % 8400 == 0:
        features = num_elements // 8400
        print(f"YOLOv11 格式 [1, {features}, 8400]")
        print(f"   predictions=8400, features={features}")
    elif num_elements % 25200 == 0:
        features = num_elements // 25200
        print(f"YOLOv5 格式 [1, 25200, {features}]")
    else:
        print(f"未知格式，元素数: {num_elements}")
    
    # 清理
    acl.mdl.destroy_desc(model_desc)
    acl.mdl.unload(model_id)
    acl.rt.destroy_context(context)
    acl.rt.reset_device(device_id)
    acl.finalize()

if __name__ == "__main__":
    check_om_model("/home/yolov11-3.0/modelv1/6_bs1.om")
