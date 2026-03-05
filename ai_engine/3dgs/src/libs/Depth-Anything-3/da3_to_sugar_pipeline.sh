#!/bin/bash
# Depth Anything 3 -> SuGaR 完整Pipeline
# 用法: ./da3_to_sugar_pipeline.sh <DA3输出目录> <场景名称> [SuGaR参数]

set -e  # 任何命令失败立即退出

# ================= 配置 =================
DA3_OUTPUT_DIR="${1:-/home/ltx/projects/Depth-Anything-3/output/sugar_streaming}"
SCENE_NAME="${2:-sugar_video}"

# SuGaR参数
REGULARIZATION="${3:-dn_consistency}"  # dn_consistency, density, sdf
REFINEMENT_TIME="${4:-short}"           # short, medium, long
HIGH_POLY="${5:-true}"                  # true/false
FAST_MODE="${6:-false}"                 # true/false

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
SUGAR_DIR="/home/ltx/projects/SuGaR"

SUGAR_DATA_DIR="$SUGAR_DIR/data/$SCENE_NAME"
SUGAR_OUTPUT_DIR="$SUGAR_DIR/output/$SCENE_NAME"

# 激活Conda环境
CONDA_ENV="gs_linux_backup"
CONDA_BASE="/home/ltx/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0

# ================= 使用说明 =================
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Depth Anything 3 -> SuGaR 完整Pipeline"
    echo ""
    echo "用法: ./da3_to_sugar_pipeline.sh <DA3输出目录> <场景名称> [SuGaR参数]"
    echo ""
    echo "参数:"
    echo "  <DA3输出目录>       Depth Anything 3的输出目录（包含camera_poses.txt等）"
    echo "  <场景名称>          SuGaR场景名称"
    echo "  [正则化方法]        dn_consistency (推荐), density, sdf"
    echo "  [精炼时间]          short (2k迭代), medium (7k), long (15k)"
    echo "  [高精度]            true (1M顶点), false (200k顶点)"
    echo "  [快速模式]          true (跳过mesh,节省50%时间), false (完整流程)"
    echo ""
    echo "示例:"
    echo "  ./da3_to_sugar_pipeline.sh ~/Depth-Anything-3/output/sugar_streaming my_scene"
    echo "  ./da3_to_sugar_pipeline.sh ~/Depth-Anything-3/output/sugar_streaming my_scene dn_consistency short true false"
    echo ""
    echo "推荐组合:"
    echo "  - 快速预览: my_scene dn_consistency short false true"
    echo "  - 标准质量: my_scene dn_consistency short true false"
    echo "  - 高质量:   my_scene dn_consistency long true false"
    exit 0
fi

if [ ! -d "$DA3_OUTPUT_DIR" ]; then
    echo "❌ 错误: DA3输出目录不存在: $DA3_OUTPUT_DIR"
    exit 1
fi

# ================= 步骤 1: 转换为COLMAP格式 =================
echo "==================== Depth Anything 3 -> SuGaR Pipeline ===================="
echo "DA3输出目录: $DA3_OUTPUT_DIR"
echo "SuGaR场景: $SCENE_NAME"
echo "正则化方法: $REGULARIZATION"
echo "精炼时间: $REFINEMENT_TIME"
echo "高精度: $HIGH_POLY"
echo "快速模式: $FAST_MODE"
echo ""

# 步骤 1.1: 生成COLMAP文本格式
echo "==== [1/4] 转换DA3输出为COLMAP格式（文本）===="

COLMAP_TEXT_DIR="$DA3_OUTPUT_DIR/colmap_text"
mkdir -p "$COLMAP_TEXT_DIR"

python3 "$DA3_DIR/convert_da3_to_colmap.py" \
    --base_dir "$DA3_OUTPUT_DIR" \
    --output_dir "$COLMAP_TEXT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 错误: DA3到COLMAP转换失败"
    exit 1
fi

echo "✅ COLMAP文本格式生成完成"

# 步骤 1.2: 转换为二进制格式
echo ""
echo "==== [2/4] 转换COLMAP文本格式为二进制格式 ===="

python3 "$DA3_DIR/colmap_text_to_binary.py" \
    "$COLMAP_TEXT_DIR/sparse/0"

if [ $? -ne 0 ]; then
    echo "❌ 错误: COLMAP二进制转换失败"
    exit 1
fi

echo "✅ COLMAP二进制格式转换完成"

# ================= 步骤 2: 整理SuGaR数据目录 =================
echo ""
echo "==== [3/4] 整理SuGaR数据目录 ===="

# 清理旧数据
if [ -d "$SUGAR_DATA_DIR" ]; then
    echo "  清理旧数据: $SUGAR_DATA_DIR"
    rm -rf "$SUGAR_DATA_DIR"
fi

# 创建新数据目录
mkdir -p "$SUGAR_DATA_DIR/sparse/0"
mkdir -p "$SUGAR_DATA_DIR/images"

# 复制COLMAP数据
echo "  复制COLMAP二进制文件..."
cp "$COLMAP_TEXT_DIR/sparse/0"/*.bin "$SUGAR_DATA_DIR/sparse/0/"

# 复制图像
echo "  复制图像..."
if [ -L "$COLMAP_TEXT_DIR/images" ]; then
    # 如果是符号链接，尝试解析目标
    LINK_TARGET="$(readlink -f "$COLMAP_TEXT_DIR/images" 2>/dev/null)"
    if [ -z "$LINK_TARGET" ] || [ ! -d "$LINK_TARGET" ]; then
        # 符号链接损坏或相对路径无法解析，直接使用已知的源目录
        echo "  ⚠️ 符号链接无法解析，使用源图像目录: $DA3_OUTPUT_DIR/extracted"
        LINK_TARGET="$DA3_OUTPUT_DIR/extracted"
    fi
    echo "  复制图像源: $LINK_TARGET -> $SUGAR_DATA_DIR/images"
    cp -r "$LINK_TARGET"/* "$SUGAR_DATA_DIR/images/"
elif [ -d "$COLMAP_TEXT_DIR/images" ]; then
    cp -r "$COLMAP_TEXT_DIR/images"/* "$SUGAR_DATA_DIR/images/"
else
    echo "  图像目录不存在: $COLMAP_TEXT_DIR/images"
    exit 1
fi

# 统计文件数量
CAMERA_COUNT=$(ls -1 "$SUGAR_DATA_DIR/sparse/0"/*.bin 2>/dev/null | wc -l)
IMAGE_COUNT=$(ls -1 "$SUGAR_DATA_DIR/images"/*.jpg "$SUGAR_DATA_DIR/images"/*.png "$SUGAR_DATA_DIR/images"/*.jpeg 2>/dev/null | wc -l)

echo "  ✅ 复制了 $CAMERA_COUNT 个COLMAP文件"
echo "  ✅ 复制了 $IMAGE_COUNT 张图像"

if [ $IMAGE_COUNT -lt 30 ]; then
    echo "  ⚠️ 警告: 图像数量较少 ($IMAGE_COUNT < 30)"
fi

# ================= 步骤 3: SuGaR训练 =================
echo ""
echo "==== [4/4] SuGaR训练 ===="

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

# 根据快速模式选择训练脚本
if [ "$FAST_MODE" = "true" ]; then
    # 快速模式
    python train_fast.py \
        -s "$SUGAR_DATA_DIR" \
        -r "$REGULARIZATION" \
        --fast_mode

    if [ $? -ne 0 ]; then
        echo "❌ 错误: SuGaR快速训练失败"
        exit 1
    fi

    echo ""
    echo "==================== ✨ 快速Pipeline完成! ===================="
    echo ""
    echo "✅ 训练完成（跳过了mesh生成和refinement）"
    echo ""
    echo "输出文件:"
    echo "  - Checkpoints & Results: $SUGAR_OUTPUT_DIR/"
    echo "  - Point Cloud: $SUGAR_OUTPUT_DIR/point_cloud/"
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 使用SuGaR查看器:"
    echo "   cd $SUGAR_DIR"
    echo "   python run_viewer.py -p $SUGAR_OUTPUT_DIR/"
    echo ""
    echo "2. 如需生成mesh，运行完整pipeline:"
    echo "   cd $SCRIPT_DIR"
    echo "   ./da3_to_sugar_pipeline.sh \"$DA3_OUTPUT_DIR\" $SCENE_NAME $REGULARIZATION $REFINEMENT_TIME $HIGH_POLY false"
    echo ""
    exit 0
else
    # 完整模式
    python train_full_pipeline.py \
        -s "$SUGAR_DATA_DIR" \
        -r "$REGULARIZATION" \
        ${HIGH_POLY:+--high_poly $HIGH_POLY} \
        --refinement_time "$REFINEMENT_TIME"

    if [ $? -ne 0 ]; then
        echo "❌ 错误: SuGaR完整训练失败"
        exit 1
    fi
fi

# ================= 完成 =================
echo ""
echo "==================== ✨ Pipeline完成! ===================="
echo ""
echo "输出文件:"

REFINED_PLY_DIR="$SUGAR_DIR/output/refined_ply/$SCENE_NAME"
REFINED_MESH_DIR="$SUGAR_DIR/output/refined_mesh/$SCENE_NAME"

# 查找生成的文件
PLY_FILE=$(ls "$REFINED_PLY_DIR"/*.ply 2>/dev/null | head -n 1)
OBJ_FILE=$(ls "$REFINED_MESH_DIR"/*.obj 2>/dev/null | head -n 1)

if [ ! -z "$PLY_FILE" ]; then
    echo "  - PLY文件 (3DGS viewer): $PLY_FILE"
fi
if [ ! -z "$OBJ_FILE" ]; then
    echo "  - OBJ文件 (传统网格): $OBJ_FILE"
fi

echo ""
echo "下一步操作:"
echo ""
echo "1. 使用SuGaR查看器:"
if [ ! -z "$PLY_FILE" ]; then
    echo "   cd $SUGAR_DIR"
    echo "   python run_viewer.py -p \"$PLY_FILE\""
fi
echo ""
echo "2. 使用其他3DGS查看器:"
echo "   - SuperSplat (在线): https://playcanvas.com/supersplat/editor"
echo "   - SuperSplat (本地): https://github.com/playcanvas/supersplat"
echo ""
echo "3. 在Blender中编辑网格:"
echo "   - 安装Blender add-on: https://github.com/Anttwo/sugar_frosting_blender_addon"
if [ ! -z "$OBJ_FILE" ]; then
    echo "   - 导入: $OBJ_FILE"
fi
echo ""
echo "4. 渲染Blender场景:"
echo "   cd $SUGAR_DIR"
echo "   python render_blender_scene.py -p <rendering_package_path>"
echo ""
