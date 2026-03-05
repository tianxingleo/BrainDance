#!/bin/bash
# Depth Anything 3 -> 纯 3DGS 训练（无SDF，无mesh，最快速度获取高质量3DGS）
#
# 用法: ./da3_to_3dgs.sh <DA3输出目录> <场景名称> [迭代次数]
#
# 对比 da3_to_sugar_pipeline.sh (dn_consistency long true false):
#   - SuGaR完整流程: Vanilla 3DGS(7k) + Coarse SuGaR(15k含SDF) + Mesh + Refinement(15k) ≈ 2-3小时
#   - 本脚本:        纯 Vanilla 3DGS(30k) ≈ 15-30分钟
#
# 输出:
#   - PLY文件: SuGaR/output/3dgs/<场景名称>/point_cloud/iteration_<N>/point_cloud.ply
#   - 可直接在 SuperSplat / splatviewer 等查看器中使用

set -e

# ================= 配置 =================
DA3_OUTPUT_DIR="${1:-output/sugar_streaming}"
SCENE_NAME="${2:-my_scene}"
ITERATIONS="${3:-30000}"  # 默认30k迭代（标准3DGS设置）

# 路径配置
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
SUGAR_DIR="/home/ltx/projects/SuGaR"
COLMAP_TEXT_DIR="$DA3_OUTPUT_DIR/colmap_text"
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
    echo "Depth Anything 3 -> 纯 3DGS 训练（快速高质量）"
    echo ""
    echo "用法: ./da3_to_3dgs.sh <DA3输出目录> <场景名称> [迭代次数]"
    echo ""
    echo "参数:"
    echo "  <DA3输出目录>   DA3的输出目录（默认: output/sugar_streaming）"
    echo "  <场景名称>      场景名称（默认: my_scene）"
    echo "  [迭代次数]      训练迭代数（默认: 30000）"
    echo ""
    echo "迭代次数参考:"
    echo "  7000   快速预览（~5分钟）"
    echo "  15000  标准质量（~10分钟）"
    echo "  30000  高质量（~20分钟，推荐）"
    echo "  50000  最高质量（~35分钟）"
    echo ""
    echo "对比 SuGaR 完整流程:"
    echo "  ./da3_to_sugar_pipeline.sh ... dn_consistency long true false  → 2-3小时（含SDF+mesh）"
    echo "  ./da3_to_3dgs.sh ... 30000                                     → 20分钟（纯3DGS）"
    echo ""
    echo "输出:"
    echo "  PLY: SuGaR/output/3dgs/<场景>/point_cloud/iteration_<N>/point_cloud.ply"
    echo ""
    echo "查看方法:"
    echo "  SuperSplat: https://playcanvas.com/supersplat/editor"
    echo "  拖拽 PLY 文件即可"
    exit 0
fi

if [ ! -d "$DA3_OUTPUT_DIR" ]; then
    echo "❌ 错误: DA3输出目录不存在: $DA3_OUTPUT_DIR"
    exit 1
fi

echo "==================== DA3 → 纯 3DGS 训练 ===================="
echo "DA3输出目录: $DA3_OUTPUT_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "输出目录: $GS_OUTPUT_DIR"
echo ""

# ================= 步骤 1: 转换为 COLMAP 格式 =================
echo "==== [1/3] 转换 DA3 → COLMAP 格式 ===="

mkdir -p "$COLMAP_TEXT_DIR"

# 检查是否已有转换结果
if [ -f "$COLMAP_TEXT_DIR/sparse/0/cameras.bin" ] && [ -f "$COLMAP_TEXT_DIR/sparse/0/images.bin" ]; then
    echo "  ✅ 已有COLMAP数据，跳过转换"
else
    # 文本格式
    python3 "$DA3_DIR/convert_da3_to_colmap.py" \
        --base_dir "$DA3_OUTPUT_DIR" \
        --output_dir "$COLMAP_TEXT_DIR"

    # 二进制格式
    python3 "$DA3_DIR/colmap_text_to_binary.py" \
        "$COLMAP_TEXT_DIR/sparse/0"

    echo "  ✅ COLMAP 转换完成"
fi

# ================= 步骤 2: 整理数据目录 =================
echo ""
echo "==== [2/3] 整理数据目录 ===="

if [ -d "$SUGAR_DATA_DIR" ]; then
    echo "  清理旧数据: $SUGAR_DATA_DIR"
    rm -rf "$SUGAR_DATA_DIR"
fi

mkdir -p "$SUGAR_DATA_DIR/sparse/0"
mkdir -p "$SUGAR_DATA_DIR/images"

# 复制 COLMAP 二进制
cp "$COLMAP_TEXT_DIR/sparse/0"/*.bin "$SUGAR_DATA_DIR/sparse/0/"

# 复制图像
LINK_TARGET="$(readlink -f "$COLMAP_TEXT_DIR/images" 2>/dev/null)"
if [ -z "$LINK_TARGET" ] || [ ! -d "$LINK_TARGET" ]; then
    LINK_TARGET="$DA3_OUTPUT_DIR/extracted"
fi
cp -r "$LINK_TARGET"/* "$SUGAR_DATA_DIR/images/"

IMAGE_COUNT=$(ls -1 "$SUGAR_DATA_DIR/images"/*.jpg "$SUGAR_DATA_DIR/images"/*.png "$SUGAR_DATA_DIR/images"/*.jpeg 2>/dev/null | wc -l)
echo "  ✅ 复制了 $IMAGE_COUNT 张图像"

# ================= 步骤 3: 训练 Vanilla 3DGS =================
echo ""
echo "==== [3/3] 训练 Vanilla 3DGS ($ITERATIONS 迭代) ===="

# 清理端口
PORT_PID=$(netstat -nlp 2>/dev/null | grep :6009 | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PORT_PID" ]; then
    echo "  清理端口 6009..."
    kill -9 $PORT_PID 2>/dev/null || true
    sleep 1
fi

cd "$SUGAR_DIR"

echo "  开始训练..."
echo ""

CUDA_VISIBLE_DEVICES=0 python ./gaussian_splatting/train.py \
    -s "$SUGAR_DATA_DIR" \
    -m "$GS_OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations $ITERATIONS \
    --test_iterations $ITERATIONS

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
