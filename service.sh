#!/bin/bash
# ============================================================
# UAV Detection gRPC 服务管理脚本
# ============================================================
# 
# 使用方法:
#   ./service.sh start    - 启动服务
#   ./service.sh stop     - 停止服务
#   ./service.sh restart  - 重启服务
#   ./service.sh status   - 查看服务状态
#   ./service.sh log      - 查看实时日志
#   ./service.sh tail     - 查看最近100行日志
#   ./service.sh logfile  - 显示日志文件路径
#   ./service.sh clean    - 清理日志文件
#
# ============================================================

# 配置区域（根据实际情况修改）
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="uav-detection"
PYTHON_SCRIPT="main_grpc_npu.py"
WORK_DIR="${SCRIPT_DIR}"
CONDA_ENV="yolov5"

# 日志配置
LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="${LOG_DIR}/${SERVICE_NAME}.log"
PID_FILE="${WORK_DIR}/${SERVICE_NAME}.pid"

# 服务端口（用于状态检查）
SERVICE_PORT=8000

# 文件描述符限制
ULIMIT_N=65535

# ============================================================
# 颜色定义
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================
# 工具函数
# ============================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 确保日志目录存在
ensure_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        log_info "创建日志目录: $LOG_DIR"
    fi
}

# 获取服务PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "$pid"
            return 0
        fi
    fi
    
    # 尝试通过进程名查找
    local pid=$(pgrep -f "$PYTHON_SCRIPT" | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    
    return 1
}

# 检查端口是否被占用
check_port() {
    if command -v ss &> /dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${SERVICE_PORT} "
    elif command -v netstat &> /dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":${SERVICE_PORT} "
    else
        return 1
    fi
}

# 激活 Conda 环境
activate_conda() {
    # 尝试常见的 conda 路径
    local conda_paths=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/root/miniconda3/etc/profile.d/conda.sh"
        "/root/anaconda3/etc/profile.d/conda.sh"
    )
    
    for conda_path in "${conda_paths[@]}"; do
        if [ -f "$conda_path" ]; then
            source "$conda_path"
            conda activate "$CONDA_ENV" 2>/dev/null
            return 0
        fi
    done
    
    # 如果已经在 conda 环境中
    if command -v conda &> /dev/null; then
        conda activate "$CONDA_ENV" 2>/dev/null
        return 0
    fi
    
    log_warn "无法找到 Conda，尝试直接使用系统 Python"
    return 1
}

# ============================================================
# 服务操作函数
# ============================================================

do_start() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  启动 $SERVICE_NAME 服务${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    # 检查是否已运行
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        log_warn "服务已在运行中 (PID: $pid)"
        return 1
    fi
    
    # 检查端口
    if check_port; then
        log_error "端口 $SERVICE_PORT 已被占用"
        return 1
    fi
    
    # 确保日志目录存在
    ensure_log_dir
    
    # 进入工作目录
    cd "$WORK_DIR" || {
        log_error "无法进入工作目录: $WORK_DIR"
        return 1
    }
    
    # 检查 Python 脚本
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "找不到 Python 脚本: $PYTHON_SCRIPT"
        return 1
    fi
    
    # 设置文件描述符限制
    ulimit -n $ULIMIT_N 2>/dev/null
    log_info "文件描述符限制: $(ulimit -n)"
    
    # 激活 Conda 环境
    activate_conda
    log_info "Python 路径: $(which python)"
    
    # 启动服务
    log_info "启动服务..."
    nohup python "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    
    # 保存 PID
    echo $new_pid > "$PID_FILE"
    
    # 等待并检查是否启动成功
    log_info "等待服务启动..."
    sleep 3
    
    if ps -p $new_pid > /dev/null 2>&1; then
        log_info "服务启动成功！"
        echo ""
        echo -e "  ${GREEN}PID:${NC}      $new_pid"
        echo -e "  ${GREEN}端口:${NC}     $SERVICE_PORT"
        echo -e "  ${GREEN}日志:${NC}     $LOG_FILE"
        echo -e "  ${GREEN}工作目录:${NC} $WORK_DIR"
        echo ""
        echo -e "  查看日志: ${YELLOW}$0 log${NC}"
        echo ""
        return 0
    else
        log_error "服务启动失败，请查看日志:"
        echo ""
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

do_stop() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  停止 $SERVICE_NAME 服务${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    local pid=$(get_pid)
    if [ -z "$pid" ]; then
        log_warn "服务未在运行"
        rm -f "$PID_FILE"
        return 0
    fi
    
    log_info "正在停止服务 (PID: $pid)..."
    
    # 先尝试优雅停止
    kill -TERM "$pid" 2>/dev/null
    
    # 等待进程退出
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""
    
    # 如果还没退出，强制杀死
    if ps -p "$pid" > /dev/null 2>&1; then
        log_warn "进程未响应，强制终止..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    
    # 清理 PID 文件
    rm -f "$PID_FILE"
    
    if ps -p "$pid" > /dev/null 2>&1; then
        log_error "无法停止服务"
        return 1
    else
        log_info "服务已停止"
        return 0
    fi
}

do_restart() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  重启 $SERVICE_NAME 服务${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    do_stop
    sleep 2
    do_start
}

do_status() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  $SERVICE_NAME 服务状态${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    local pid=$(get_pid)
    
    if [ -n "$pid" ]; then
        echo -e "  状态:     ${GREEN}运行中${NC}"
        echo -e "  PID:      $pid"
        
        # 获取进程详细信息
        local cpu_mem=$(ps -p "$pid" -o %cpu,%mem,etime --no-headers 2>/dev/null)
        if [ -n "$cpu_mem" ]; then
            local cpu=$(echo $cpu_mem | awk '{print $1}')
            local mem=$(echo $cpu_mem | awk '{print $2}')
            local uptime=$(echo $cpu_mem | awk '{print $3}')
            echo -e "  CPU:      ${cpu}%"
            echo -e "  内存:     ${mem}%"
            echo -e "  运行时间: $uptime"
        fi
        
        # 检查端口
        if check_port; then
            echo -e "  端口:     ${GREEN}$SERVICE_PORT (监听中)${NC}"
        else
            echo -e "  端口:     ${YELLOW}$SERVICE_PORT (未监听)${NC}"
        fi
        
        echo -e "  日志:     $LOG_FILE"
        echo ""
        
        # 显示最近的日志
        echo -e "${BLUE}最近日志:${NC}"
        echo "----------------------------------------"
        if [ -f "$LOG_FILE" ]; then
            tail -5 "$LOG_FILE"
        else
            echo "(无日志文件)"
        fi
        echo "----------------------------------------"
        
    else
        echo -e "  状态: ${RED}未运行${NC}"
        
        if check_port; then
            echo -e "  ${YELLOW}警告: 端口 $SERVICE_PORT 被其他进程占用${NC}"
        fi
    fi
    echo ""
}

do_log() {
    if [ ! -f "$LOG_FILE" ]; then
        log_error "日志文件不存在: $LOG_FILE"
        return 1
    fi
    
    echo ""
    echo -e "${CYAN}实时日志 (Ctrl+C 退出)${NC}"
    echo -e "${CYAN}文件: $LOG_FILE${NC}"
    echo "========================================"
    tail -f "$LOG_FILE"
}

do_tail() {
    local lines=${1:-100}
    
    if [ ! -f "$LOG_FILE" ]; then
        log_error "日志文件不存在: $LOG_FILE"
        return 1
    fi
    
    echo ""
    echo -e "${CYAN}最近 $lines 行日志${NC}"
    echo -e "${CYAN}文件: $LOG_FILE${NC}"
    echo "========================================"
    tail -n "$lines" "$LOG_FILE"
}

do_logfile() {
    echo ""
    echo -e "${CYAN}日志文件信息${NC}"
    echo "========================================"
    echo -e "路径: ${GREEN}$LOG_FILE${NC}"
    
    if [ -f "$LOG_FILE" ]; then
        local size=$(du -h "$LOG_FILE" | cut -f1)
        local lines=$(wc -l < "$LOG_FILE")
        echo -e "大小: $size"
        echo -e "行数: $lines"
        echo -e "修改时间: $(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null)"
    else
        echo -e "${YELLOW}(文件不存在)${NC}"
    fi
    echo ""
}

do_clean() {
    echo ""
    echo -e "${CYAN}清理日志文件${NC}"
    echo "========================================"
    
    if [ -f "$LOG_FILE" ]; then
        local size=$(du -h "$LOG_FILE" | cut -f1)
        read -p "确定要清理日志文件吗？当前大小: $size [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # 备份最近的日志
            local backup="${LOG_FILE}.$(date +%Y%m%d_%H%M%S).bak"
            tail -1000 "$LOG_FILE" > "$backup"
            > "$LOG_FILE"
            log_info "日志已清理，最近1000行已备份到: $backup"
        else
            log_info "取消清理"
        fi
    else
        log_info "日志文件不存在，无需清理"
    fi
}

do_help() {
    echo ""
    echo -e "${CYAN}$SERVICE_NAME 服务管理脚本${NC}"
    echo ""
    echo "使用方法: $0 <命令> [参数]"
    echo ""
    echo "命令列表:"
    echo -e "  ${GREEN}start${NC}     启动服务"
    echo -e "  ${GREEN}stop${NC}      停止服务"
    echo -e "  ${GREEN}restart${NC}   重启服务"
    echo -e "  ${GREEN}status${NC}    查看服务状态"
    echo -e "  ${GREEN}log${NC}       查看实时日志 (tail -f)"
    echo -e "  ${GREEN}tail${NC} [n]  查看最近n行日志 (默认100行)"
    echo -e "  ${GREEN}logfile${NC}   显示日志文件信息"
    echo -e "  ${GREEN}clean${NC}     清理日志文件"
    echo -e "  ${GREEN}help${NC}      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start          # 启动服务"
    echo "  $0 log            # 查看实时日志"
    echo "  $0 tail 50        # 查看最近50行日志"
    echo "  $0 status         # 查看服务状态"
    echo ""
    echo "配置:"
    echo "  工作目录: $WORK_DIR"
    echo "  Conda环境: $CONDA_ENV"
    echo "  服务端口: $SERVICE_PORT"
    echo "  日志目录: $LOG_DIR"
    echo ""
}

# ============================================================
# 主入口
# ============================================================
case "$1" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        do_status
        ;;
    log)
        do_log
        ;;
    tail)
        do_tail "$2"
        ;;
    logfile)
        do_logfile
        ;;
    clean)
        do_clean
        ;;
    help|--help|-h)
        do_help
        ;;
    *)
        if [ -n "$1" ]; then
            log_error "未知命令: $1"
        fi
        do_help
        exit 1
        ;;
esac
