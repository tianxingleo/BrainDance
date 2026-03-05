#!/bin/bash
# DA3 3DGS 直接启动脚本（前台运行）

set -e

# 配置
MODEL_DIR="./weights"
WORKSPACE_DIR="./workspace/da3_3dgs"
GALLERY_DIR="./gallery/da3_3dgs"
HOST="127.0.0.1"
PORT=7860

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup

echo "============================================================"
echo "DA3 3DGS 生成工具（前台模式）"
echo "============================================================"
echo "访问地址：http://$HOST:$PORT"
echo "============================================================"
echo ""
echo "⚠️  按 Ctrl+C 停止应用"
echo ""

# 创建目录
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$GALLERY_DIR"

# 启动 Gradio 应用（前台）
cd /home/ltx/projects/Depth-Anything-3
python -m depth_anything_3.app.gradio_app \
    --model-dir "$MODEL_DIR" \
    --workspace-dir "$WORKSPACE_DIR" \
    --gallery-dir "$GALLERY_DIR" \
    --host "$HOST" \
    --port "$PORT"
