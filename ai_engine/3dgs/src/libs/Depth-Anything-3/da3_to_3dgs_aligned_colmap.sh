#!/bin/bash
# 方案 A: Depth Anything 3 -> COLMAP 平面对齐 -> 纯 3DGS 训练
#
# 基于 da3_to_3dgs.sh，增加了 COLMAP model_aligner 步骤
# 利用 COLMAP 原生命令的"曼哈顿世界假设"自动检测地面并对齐到 X-Y 平面
#
# 用法: ./da3_to_3dgs_aligned_colmap.sh <DA3输出目录> <场景名称> [迭代次数] [对齐误差阈值]
#
# 对比 da3_to_3dgs.sh:
#   - 增加了 COLMAP model_aligner 步骤（在转换后、训练前）
#   - 3DGS 训练过程完全一致
#
# 优点: 最简单，无需额外依赖，COLMAP 自带
# 缺点: 依赖 COLMAP 的平面检测算法质量

set -e

# ================= 配置 =================
DA3_OUTPUT_DIR="${1:-output/sugar_streaming}"
SCENE_NAME="${2:-my_scene}"
ITERATIONS="${3:-30000}"          # 默认30k迭代（标准3DGS设置）
ALIGNMENT_MAX_ERROR="${4:-0.02}"  # COLMAP对齐最大误差 (米)

# 路径配置
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
SUGAR_DIR="/home/ltx/projects/SuGaR"
COLMAP_TEXT_DIR="$DA3_OUTPUT_DIR/colmap_text"
COLMAP_ALIGNED_DIR="$DA3_OUTPUT_DIR/colmap_text/sparse/aligned"  # 对齐后的模型
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
    echo "方案 A: Depth Anything 3 -> COLMAP 对齐 -> 纯 3DGS 训练"
    echo ""
    echo "用法: ./da3_to_3dgs_aligned_colmap.sh <DA3输出目录> <场景名称> [迭代次数] [对齐误差阈值]"
    echo ""
    echo "参数:"
    echo "  <DA3输出目录>     DA3的输出目录（默认: output/sugar_streaming）"
    echo "  <场景名称>        场景名称（默认: my_scene）"
    echo "  [迭代次数]        训练迭代数（默认: 30000）"
    echo "  [对齐误差阈值]    COLMAP对齐误差(米)（默认: 0.02）"
    echo ""
    echo "原理:"
    echo "  使用 COLMAP model_aligner --alignment_type plane"
    echo "  自动利用曼哈顿世界假设检测主平面（通常是地面）"
    echo "  将模型旋转使地面对齐到 X-Y 平面"
    echo ""
    echo "与 da3_to_3dgs.sh 的区别:"
    echo "  仅在步骤1后增加了 COLMAP model_aligner 对齐步骤"
    echo "  训练过程完全一致"
    echo ""
    echo "输出:"
    echo "  PLY: SuGaR/output/3dgs/<场景>/point_cloud/iteration_<N>/point_cloud.ply"
    exit 0
fi

if [ ! -d "$DA3_OUTPUT_DIR" ]; then
    echo "❌ 错误: DA3输出目录不存在: $DA3_OUTPUT_DIR"
    exit 1
fi

echo "==================== DA3 → COLMAP 对齐 → 纯 3DGS 训练 (方案A) ===================="
echo "DA3输出目录: $DA3_OUTPUT_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "对齐误差阈值: $ALIGNMENT_MAX_ERROR"
echo "输出目录: $GS_OUTPUT_DIR"
echo ""

# ================= 步骤 1: 转换为 COLMAP 格式 =================
echo "==== [1/4] 转换 DA3 → COLMAP 格式 ===="

mkdir -p "$COLMAP_TEXT_DIR"

# 检查是否已有转换结果（未对齐的原始版本）
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

# ================= 步骤 2: COLMAP 平面对齐 =================
echo ""
echo "==== [2/4] COLMAP 平面对齐 (model_aligner) ===="

# 创建对齐输出目录
mkdir -p "$COLMAP_ALIGNED_DIR"

echo "  输入模型: $COLMAP_TEXT_DIR/sparse/0"
echo "  输出模型: $COLMAP_ALIGNED_DIR"
echo "  对齐方式: plane (曼哈顿世界假设)"
echo "  最大误差: $ALIGNMENT_MAX_ERROR"
echo ""

# 运行 COLMAP model_aligner
colmap model_aligner \
    --input_path "$COLMAP_TEXT_DIR/sparse/0" \
    --output_path "$COLMAP_ALIGNED_DIR" \
    --ref_is_gps 0 \
    --alignment_type plane \
    --alignment_max_error "$ALIGNMENT_MAX_ERROR"

if [ $? -eq 0 ]; then
    echo "  ✅ COLMAP 平面对齐完成"
    # 验证对齐后的文件
    if [ -f "$COLMAP_ALIGNED_DIR/cameras.bin" ] && [ -f "$COLMAP_ALIGNED_DIR/images.bin" ]; then
        echo "  ✅ 对齐后的模型文件已生成"
        # 使用对齐后的模型
        USE_ALIGNED=true
    else
        echo "  ⚠️ 对齐后的模型文件缺失，回退使用原始模型"
        USE_ALIGNED=false
    fi
else
    echo "  ⚠️ COLMAP 对齐失败，回退使用原始未对齐模型"
    USE_ALIGNED=false
fi

# ================= 步骤 3: 整理数据目录 =================
echo ""
echo "==== [3/4] 整理数据目录 ===="

if [ -d "$SUGAR_DATA_DIR" ]; then
    echo "  清理旧数据: $SUGAR_DATA_DIR"
    rm -rf "$SUGAR_DATA_DIR"
fi

mkdir -p "$SUGAR_DATA_DIR/sparse/0"
mkdir -p "$SUGAR_DATA_DIR/images"

# 复制 COLMAP 二进制 (优先使用对齐后的)
if [ "$USE_ALIGNED" = true ]; then
    echo "  📐 使用 COLMAP 对齐后的模型"
    cp "$COLMAP_ALIGNED_DIR"/*.bin "$SUGAR_DATA_DIR/sparse/0/"
else
    echo "  ⚠️ 使用原始未对齐的模型"
    cp "$COLMAP_TEXT_DIR/sparse/0"/*.bin "$SUGAR_DATA_DIR/sparse/0/"
fi

# 复制图像
LINK_TARGET="$(readlink -f "$COLMAP_TEXT_DIR/images" 2>/dev/null)"
if [ -z "$LINK_TARGET" ] || [ ! -d "$LINK_TARGET" ]; then
    LINK_TARGET="$DA3_OUTPUT_DIR/extracted"
fi
cp -r "$LINK_TARGET"/* "$SUGAR_DATA_DIR/images/"

IMAGE_COUNT=$(ls -1 "$SUGAR_DATA_DIR/images"/*.jpg "$SUGAR_DATA_DIR/images"/*.png "$SUGAR_DATA_DIR/images"/*.jpeg 2>/dev/null | wc -l)
echo "  ✅ 复制了 $IMAGE_COUNT 张图像"

# ================= 步骤 4: 训练 Vanilla 3DGS =================
echo ""
echo "==== [4/4] 训练 Vanilla 3DGS ($ITERATIONS 迭代) ===="

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
echo "==================== ✨ 训练完成! (方案A: COLMAP对齐) ===================="
echo ""

# 查找输出 PLY
PLY_FILE="$GS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
if [ -f "$PLY_FILE" ]; then
    PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
    echo "输出文件:"
    echo "  PLY: $PLY_FILE ($PLY_SIZE)"
    echo ""
    if [ "$USE_ALIGNED" = true ]; then
        echo "  📐 此模型已经过 COLMAP 平面对齐 (地面 → X-Y 平面)"
    else
        echo "  ⚠️ 此模型未经对齐 (COLMAP对齐步骤失败)"
    fi
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
