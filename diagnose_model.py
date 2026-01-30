#!/usr/bin/env python3
"""
诊断 OM 模型输出结构
"""
import numpy as np
import cv2
import acl

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

def init_acl(device_id=0):
    """初始化 ACL"""
    acl.init()
    acl.rt.set_device(device_id)
    context, _ = acl.rt.create_context(device_id)
    return context

def load_model(model_path):
    """加载模型并获取详细信息"""
    model_id, ret = acl.mdl.load_from_file(model_path)
    assert ret == 0
    
    model_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(model_desc, model_id)
    
    print(f"{'='*60}")
    print(f"模型信息: {model_path}")
    print(f"{'='*60}")
    
    # 输入信息
    num_inputs = acl.mdl.get_num_inputs(model_desc)
    print(f"\n输入数量: {num_inputs}")
    for i in range(num_inputs):
        input_size = acl.mdl.get_input_size_by_index(model_desc, i)
        print(f"  输入[{i}] 大小: {input_size} bytes ({input_size/1024:.1f} KB)")
    
    # 输出信息
    num_outputs = acl.mdl.get_num_outputs(model_desc)
    print(f"\n输出数量: {num_outputs}")
    for i in range(num_outputs):
        output_size = acl.mdl.get_output_size_by_index(model_desc, i)
        print(f"  输出[{i}] 大小: {output_size} bytes ({output_size/1024:.1f} KB)")
        
        # 尝试猜测可能的形状
        num_floats_fp32 = output_size // 4
        num_floats_fp16 = output_size // 2
        print(f"    - 如果是 FP32: {num_floats_fp32} 个元素")
        print(f"    - 如果是 FP16: {num_floats_fp16} 个元素")
        
        # 常见的 YOLOv5 输出形状
        possible_shapes = []
        
        # FP32
        if num_floats_fp32 % 85 == 0:
            possible_shapes.append(f"FP32: ({num_floats_fp32//85}, 85)")
        if num_floats_fp32 % 6 == 0:
            possible_shapes.append(f"FP32: ({num_floats_fp32//6}, 6)")
        
        # FP16
        if num_floats_fp16 % 85 == 0:
            possible_shapes.append(f"FP16: ({num_floats_fp16//85}, 85)")
        if num_floats_fp16 % 6 == 0:
            possible_shapes.append(f"FP16: ({num_floats_fp16//6}, 6)")
        
        if possible_shapes:
            print(f"    可能的格式: {', '.join(possible_shapes)}")
    
    return model_id, model_desc

def run_inference(model_id, model_desc, image_path):
    """运行推理并分析所有输出"""
    # 预处理
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    
    scale = min(640 / orig_h, 640 / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    img_padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    img_padded[:new_h, :new_w] = img_resized
    
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_norm = img_chw.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    img_batch = np.ascontiguousarray(img_batch)
    
    # 准备输入
    input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
    input_buffer, _ = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
    
    input_bytes = img_batch.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    acl.rt.memcpy(input_buffer, input_size, input_ptr, len(input_bytes), ACL_MEMCPY_HOST_TO_DEVICE)
    
    # 创建输入 dataset
    input_dataset = acl.mdl.create_dataset()
    input_data_buffer = acl.create_data_buffer(input_buffer, input_size)
    acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)
    
    # 准备所有输出
    num_outputs = acl.mdl.get_num_outputs(model_desc)
    output_buffers = []
    output_sizes = []
    output_data_buffers = []
    
    output_dataset = acl.mdl.create_dataset()
    
    for i in range(num_outputs):
        output_size = acl.mdl.get_output_size_by_index(model_desc, i)
        output_buffer, _ = acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST)
        output_data_buffer = acl.create_data_buffer(output_buffer, output_size)
        acl.mdl.add_dataset_buffer(output_dataset, output_data_buffer)
        
        output_buffers.append(output_buffer)
        output_sizes.append(output_size)
        output_data_buffers.append(output_data_buffer)
    
    # 推理
    print(f"\n{'='*60}")
    print(f"执行推理...")
    print(f"{'='*60}")
    
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    assert ret == 0
    
    # 获取所有输出
    outputs = []
    for i in range(num_outputs):
        output_size = output_sizes[i]
        
        # 尝试 FP32
        output_fp32 = np.zeros(output_size // 4, dtype=np.float32)
        output_bytes = output_fp32.tobytes()
        output_ptr = acl.util.bytes_to_ptr(output_bytes)
        acl.rt.memcpy(output_ptr, len(output_bytes), output_buffers[i], output_size, ACL_MEMCPY_DEVICE_TO_HOST)
        output_fp32 = np.frombuffer(output_bytes, dtype=np.float32)
        
        # 尝试 FP16
        output_fp16 = np.zeros(output_size // 2, dtype=np.float16)
        output_bytes_fp16 = output_fp16.tobytes()
        output_ptr_fp16 = acl.util.bytes_to_ptr(output_bytes_fp16)
        acl.rt.memcpy(output_ptr_fp16, len(output_bytes_fp16), output_buffers[i], output_size, ACL_MEMCPY_DEVICE_TO_HOST)
        output_fp16 = np.frombuffer(output_bytes_fp16, dtype=np.float16)
        
        print(f"\n输出 [{i}]:")
        print(f"  FP32 解析:")
        print(f"    形状: {output_fp32.shape}")
        print(f"    范围: [{output_fp32.min():.2f}, {output_fp32.max():.2f}]")
        print(f"    均值: {output_fp32.mean():.2f}")
        print(f"    非零数: {np.count_nonzero(output_fp32)}")
        
        print(f"  FP16 解析:")
        print(f"    形状: {output_fp16.shape}")
        print(f"    范围: [{float(output_fp16.min()):.2f}, {float(output_fp16.max()):.2f}]")
        print(f"    均值: {float(output_fp16.mean()):.2f}")
        print(f"    非零数: {np.count_nonzero(output_fp16)}")
        
        # 检查哪个更合理
        fp32_reasonable = np.all(np.isfinite(output_fp32)) and abs(output_fp32.max()) < 1e6
        fp16_reasonable = np.all(np.isfinite(output_fp16)) and abs(float(output_fp16.max())) < 1e6
        
        print(f"  推荐格式: {'FP32' if fp32_reasonable else 'FP16' if fp16_reasonable else '未知'}")
        
        outputs.append({
            'fp32': output_fp32,
            'fp16': output_fp16,
            'size': output_size
        })
    
    # 清理
    acl.destroy_data_buffer(input_data_buffer)
    for buf in output_data_buffers:
        acl.destroy_data_buffer(buf)
    acl.mdl.destroy_dataset(input_dataset)
    acl.mdl.destroy_dataset(output_dataset)
    acl.rt.free(input_buffer)
    for buf in output_buffers:
        acl.rt.free(buf)
    
    return outputs

def main():
    import sys
    if len(sys.argv) < 3:
        print("用法: python3 diagnose_model.py <model.om> <image.jpg>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # 初始化
    context = init_acl(0)
    
    # 加载模型
    model_id, model_desc = load_model(model_path)
    
    # 运行推理
    outputs = run_inference(model_id, model_desc, image_path)
    
    print(f"\n{'='*60}")
    print(f"诊断完成")
    print(f"{'='*60}")
    
    # 清理
    acl.mdl.destroy_desc(model_desc)
    acl.mdl.unload(model_id)
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()

if __name__ == "__main__":
    main()
