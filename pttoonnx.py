import os
import subprocess

for pt in os.listdir('.'):
    if '720' in pt and pt.endswith('.pt'):
        cmd = f"yolo export model={pt} format=onnx imgsz=720 opset=12 simplify=True batch=1"
        print(f"导出: {pt}")
        subprocess.run(cmd, shell=True)