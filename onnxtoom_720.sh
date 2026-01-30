source /etc/profile.d/ascend.sh
conda activate yolov11

for onnx in *720*.onnx; do
    if [ -f "$onnx" ]; then
        name="${onnx%.onnx}"
        echo "转换: $onnx -> ${name}.om"
        atc \
            --model="$onnx" \
            --framework=5 \
            --output="$name" \
            --input_format=NCHW \
            --input_shape="images:1,3,736,736" \
            --soc_version=Ascend310P3 \
            --output_type=FP16 \
            --log=error
        echo "完成: ${name}.om"
        echo "---"
    fi
done

echo "全部完成"
ls -la *720*.om
