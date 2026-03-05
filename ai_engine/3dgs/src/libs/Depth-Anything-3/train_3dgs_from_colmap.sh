#!/bin/bash
# 使用已有COLMAP数据进行传统3DGS训练
#
# 用法: ./train_3dgs_from_colmap.sh <COLMAP_DIR> <SCENE_NAME> [ITERATIONS]
#
# 示例: ./train_3dgs_from_colmap.sh output/sugar_streaming1_colmap sugar_streaming1 30000

set -e

# ================= 配置 =================
COLMAP_DIR="${1:-output/sugar_streaming1_colmap}"
SCENE_NAME="${2:-sugar_streaming1}"
ITERATIONS="${3:-30000}"

# 路径配置
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
SUGAR_DIR="/home/ltx/projects/SuGaR"
SUGAR_DATA_DIR="$SUGAR_DIR/data/$SCENE_NAME"
GS_OUTPUT_DIR="$SUGAR_DIR/output/3dgs/$SCENE_NAME"

# 激活环境
CONDA_ENV="gs_linux_backup"
CONDA_BASE="/home/ltx/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0

# ================= 帮助 =================
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "使用已有COLMAP数据进行传统3DGS训练"
    echo ""
    echo "用法: ./train_3dgs_from_colmap.sh <COLMAP_DIR> <SCENE_NAME> [ITERATIONS]"
    echo ""
    echo "参数:"
    echo "  <COLMAP_DIR>   COLMAP输出目录（默认: output/sugar_streaming1_colmap）"
    echo "  <SCENE_NAME>   场景名称（默认: sugar_streaming1）"
    echo "  [ITERATIONS]   训练迭代数（默认: 30000）"
    echo ""
    echo "要求:"
    echo "  - COLMAP目录包含 sparse/0/ 目录"
    echo "  - sparse/0/ 中包含 cameras.txt, images.txt, points3D.txt（文本格式）"
    echo "  - 或者包含 .bin 文件（二进制格式）"
    echo ""
    echo "输出:"
    echo "  PLY: $SUGAR_DIR/output/3dgs/<SCENE_NAME>/point_cloud/iteration_<N>/point_cloud.ply"
    exit 0
fi

if [ ! -d "$COLMAP_DIR" ]; then
    echo "❌ 错误: COLMAP目录不存在: $COLMAP_DIR"
    exit 1
fi

echo "==================== 传统 3DGS 训练 ===================="
echo "COLMAP目录: $COLMAP_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "输出目录: $GS_OUTPUT_DIR"
echo ""

# ================= 步骤 1: 检查并转换COLMAP格式 =================
echo "==== [1/3] 检查COLMAP格式 ===="

COLMAP_SPARSE_DIR="$COLMAP_DIR/sparse/0"

if [ ! -d "$COLMAP_SPARSE_DIR" ]; then
    echo "❌ 错误: COLMAP sparse目录不存在: $COLMAP_SPARSE_DIR"
    exit 1
fi

# 检查是文本格式还是二进制格式
if [ -f "$COLMAP_SPARSE_DIR/cameras.bin" ] && [ -f "$COLMAP_SPARSE_DIR/images.bin" ]; then
    echo "  ✅ 检测到二进制格式COLMAP数据"
    USE_BINARY=true
elif [ -f "$COLMAP_SPARSE_DIR/cameras.txt" ] && [ -f "$COLMAP_SPARSE_DIR/images.txt" ]; then
    echo "  ℹ️  检测到文本格式COLMAP数据，需要转换为二进制格式"
    USE_BINARY=false

    # 转换为二进制
    echo "  转换中..."
    python3 "$DA3_DIR/colmap_text_to_binary.py" "$COLMAP_SPARSE_DIR"

    if [ $? -eq 0 ]; then
        echo "  ✅ 转换完成"
        USE_BINARY=true
    else
        echo "❌ 转换失败"
        exit 1
    fi
else
    echo "❌ 错误: 未找到有效的COLMAP数据文件"
    echo "   期望: cameras.bin/images.bin 或 cameras.txt/images.txt"
    exit 1
fi

# ================= 步骤 2: 准备SuGaR数据目录 =================
echo ""
echo "==== [2/3] 准备训练数据 ===="

if [ -d "$SUGAR_DATA_DIR" ]; then
    echo "  清理旧数据: $SUGAR_DATA_DIR"
    rm -rf "$SUGAR_DATA_DIR"
fi

mkdir -p "$SUGAR_DATA_DIR/sparse/0"
mkdir -p "$SUGAR_DATA_DIR/images"

# 复制COLMAP二进制文件
echo "  复制COLMAP数据..."
cp "$COLMAP_SPARSE_DIR"/*.bin "$SUGAR_DATA_DIR/sparse/0/"
echo "  ✅ COLMAP数据已复制"

# 处理图像
if [ -L "$COLMAP_DIR/images" ]; then
    # 如果是软链接，解析其目标
    LINK_TARGET="$(readlink -f "$COLMAP_DIR/images")"
    echo "  检测到图像软链接: $LINK_TARGET"
    cp -r "$LINK_TARGET"/* "$SUGAR_DATA_DIR/images/"
elif [ -d "$COLMAP_DIR/images" ]; then
    echo "  复制图像..."
    cp -r "$COLMAP_DIR"/images/* "$SUGAR_DATA_DIR/images/"
else
    echo "❌ 错误: 未找到图像目录"
    exit 1
fi

IMAGE_COUNT=$(ls -1 "$SUGAR_DATA_DIR/images"/*.jpg "$SUGAR_DATA_DIR/images"/*.png "$SUGAR_DATA_DIR/images"/*.jpeg 2>/dev/null | wc -l)
echo "  ✅ 复制了 $IMAGE_COUNT 张图像"

# 检查并显示图片分辨率和相机参数
if [ $IMAGE_COUNT -gt 0 ]; then
    FIRST_IMG=$(ls "$SUGAR_DATA_DIR/images"/*.jpg "$SUGAR_DATA_DIR/images"/*.png "$SUGAR_DATA_DIR/images"/*.jpeg 2>/dev/null | head -1)
    if [ -n "$FIRST_IMG" ]; then
        IMG_RES=$(python3 -c "from PIL import Image; img = Image.open('$FIRST_IMG'); print(f'{img.width} x {img.height}')" 2>/dev/null)
        if [ -n "$IMG_RES" ]; then
            echo "  📷 图片分辨率: $IMG_RES"
        fi
    fi

    # 读取COLMAP相机参数（如果存在文本格式）
    CAMERAS_TXT="$COLMAP_DIR/sparse/0/cameras.txt"
    if [ -f "$CAMERAS_TXT" ]; then
        CAM_INFO=$(grep -v '^#' "$CAMERAS_TXT" | head -1 | awk -F' ' '{print "fx="$4", fy="$5", cx="$6", cy="$7}')
        if [ -n "$CAM_INFO" ]; then
            echo "  📐 COLMAP内参: $CAM_INFO"
        fi
    fi
fi

echo ""
echo "  🎞️  训练配置: $IMAGE_COUNT 张图像, $ITERATIONS 次迭代"
echo "  🎚️  高斯球控制: densify_until=$((ITERATIONS - 3000)), grad_threshold=0.0004"
echo ""

# ================= 步骤 3: 训练传统3DGS =================
echo ""
echo "==== [3/3] 训练传统 3DGS ($ITERATIONS 迭代) ===="

# 清理端口
PORT_PID=$(netstat -nlp 2>/dev/null | grep :6009 | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PORT_PID" ]; then
    echo "  清理端口 6009..."
    kill -9 $PORT_PID 2>/dev/null || true
    sleep 1
fi

cd "$SUGAR_DIR"

echo "  开始训练..."
echo "  高斯椭球控制: densify_until_iter=$((ITERATIONS - 3000)), densify_grad_threshold=0.0004"
echo ""

CUDA_VISIBLE_DEVICES=0 python ./gaussian_splatting/train.py \
    -s "$SUGAR_DATA_DIR" \
    -m "$GS_OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations $ITERATIONS \
    --test_iterations $ITERATIONS \
    --densify_until_iter $((ITERATIONS - 3000)) \
    --densify_grad_threshold 0.0004

echo ""
echo "==================== ✨ 训练完成! ===================="
echo ""

# 查找输出 PLY
PLY_FILE="$GS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
if [ -f "$PLY_FILE" ]; then
    PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
    echo "输出文件:"
    echo "  PLY: $PLY_FILE ($PLY_SIZE)"
    echo ""
    echo "查看方法:"
    echo "  1. SuperSplat (在线): https://playcanvas.com/supersplat/editor"
    echo "     拖拽 PLY 文件即可"
    echo ""
    echo "  2. 本地渲染:"
    echo "     cd $SUGAR_DIR"
    echo "     python gaussian_splatting/render.py -m $GS_OUTPUT_DIR --iteration $ITERATIONS"
    echo ""
    echo "  3. SIBR 查看器:"
    echo "     cd $SUGAR_DIR/gaussian_splatting/SIBR_viewers/install/bin"
    echo "     ./SIBR_gaussianViewer_app -m $GS_OUTPUT_DIR"
else
    echo "⚠️ 未找到 PLY 文件，请检查训练日志"
    ls -la "$GS_OUTPUT_DIR/point_cloud/" 2>/dev/null || echo "  point_cloud 目录不存在"
fi
echo ""
