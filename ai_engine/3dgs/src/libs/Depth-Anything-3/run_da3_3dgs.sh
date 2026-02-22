#!/bin/bash
# DA3 3DGS 生成脚本
# 使用 DA3 Gradio UI 生成 3D Gaussian Splatting

set -e

# 配置
IMAGE_DIR="/home/ltx/projects/SuGaR/video.mp4"  # 可以是视频或图像目录
WORKSPACE_DIR="./workspace/da3_3dgs"
GALLERY_DIR="./gallery/da3_3dgs"
MODEL_DIR="./weights"
HOST="0.0.0.0"
PORT=7860

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup

echo "============================================================"
echo "DA3 3DGS 生成工具"
echo "============================================================"
echo "模型目录: $MODEL_DIR"
echo "工作目录: $WORKSPACE_DIR"
echo "访问地址: http://$HOST:$PORT"
echo "============================================================"

# 创建目录
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$GALLERY_DIR"

# 启动 Gradio 应用
echo ""
echo "🚀 启动 Gradio 应用..."
echo "💡 在浏览器中打开应用，然后："
echo "   1. 上传图像文件夹或视频"
echo "   2. 等待处理完成"
echo "   3. 勾选 'Export 3DGS Video' 选项"
echo "   4. 选择视频质量 (low/medium/high)"
echo "   5. 点击 'Process' 按钮"
echo ""

da3 gradio \
    --model-dir "$MODEL_DIR" \
    --workspace-dir "$WORKSPACE_DIR" \
    --gallery-dir "$GALLERY_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --share
