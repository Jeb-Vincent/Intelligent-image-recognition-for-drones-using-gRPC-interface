@echo off
chcp 65001 >nul
title NPU 目标检测演示服务

echo ============================================================
echo   NPU 目标检测演示系统
echo   (通过 gRPC 连接华为 NPU 服务器)
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
echo [检查] 正在检查依赖...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装依赖...
    pip install flask flask-cors grpcio grpcio-tools opencv-python numpy Pillow protobuf
)

echo.
echo [提示] 配置在 npu_demo_server.py 顶部修改
echo [提示] 默认 NPU 服务器: 172.18.8.11:8000
echo [提示] 按 Ctrl+C 停止服务
echo ============================================================
echo.

REM 启动服务
python npu_demo_server.py

pause
