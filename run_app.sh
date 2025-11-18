#!/bin/bash
# Convenience launcher for the handwritten digit & formula recognizer GUI.
#
# Usage:
#     ./run_app.sh             # ensure model exists, warn about pix2tex, then launch GUI
#     ./run_app.sh --force-train
#     ./run_app.sh --train-only

set -e  # Exit on error

# Get the directory where this script is located
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$ROOT/model_scripted.pt"
STATE_MODEL="$ROOT/model_state.pt"

# Find Python executable (prefer the one from anaconda if available)
if command -v /opt/anaconda3/bin/python &> /dev/null; then
    PYTHON="/opt/anaconda3/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "[run_app] 错误: 未找到 Python 解释器"
    exit 1
fi

# Parse command line arguments
FORCE_TRAIN=false
TRAIN_ONLY=false

for arg in "$@"; do
    case $arg in
        --force-train)
            FORCE_TRAIN=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        *)
            echo "未知参数: $arg"
            echo "用法: $0 [--force-train] [--train-only]"
            exit 1
            ;;
    esac
done

# Function to run a command
_run_command() {
    echo "[run_app] 执行: $*"
    "$@"
}

# Ensure model exists
ensure_model() {
    if [ "$FORCE_TRAIN" = true ] || { [ ! -f "$MODEL_PATH" ] && [ ! -f "$STATE_MODEL" ]; }; then
        echo "[run_app] 未找到模型或指定重新训练，开始训练 ..."
        _run_command "$PYTHON" train_model.py --no-plot
        return 0
    else
        return 1
    fi
}

# Launch GUI
launch_gui() {
    echo "[run_app] 启动 GUI ..."
    _run_command "$PYTHON" app_gui.py
}

# Main execution
cd "$ROOT"

# Train model if needed
if ensure_model; then
    MODEL_TRAINED=true
else
    MODEL_TRAINED=false
fi

# If train-only, exit after training
if [ "$TRAIN_ONLY" = true ]; then
    echo "[run_app] 训练完成，已按要求退出。"
    exit 0
fi

# If model wasn't trained, inform user
if [ "$MODEL_TRAINED" = false ]; then
    if [ -f "$MODEL_PATH" ]; then
        echo "[run_app] 已检测到现有模型 (model_scripted.pt)，跳过训练。"
    elif [ -f "$STATE_MODEL" ]; then
        echo "[run_app] 已检测到 State Dict 模型 (model_state.pt)，跳过训练。"
    else
        echo "[run_app] 未检测到模型文件，但训练步骤被跳过。请手动检查。"
    fi
fi

# Launch GUI
launch_gui
