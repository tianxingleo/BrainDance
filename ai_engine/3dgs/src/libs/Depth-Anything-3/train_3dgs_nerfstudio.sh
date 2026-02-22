#!/bin/bash
# 使用 Nerfstudio 的 splatfacto 进行 3DGS 训练
# 基于 run_da3_to_3dgs_aligned.py 的逻辑
#
# 用法: ./train_3dgs_nerfstudio.sh <COLMAP_DIR> <SCENE_NAME> [ITERATIONS]

set -e

# ================= 配置 =================
COLMAP_DIR="${1:-output/sugar_streaming1_colmap}"
SCENE_NAME="${2:-sugar_streaming1}"
ITERATIONS="${3:-15000}"

# 路径配置
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
CONDA_PREFIX="/home/ltx/my_envs/gs_linux_backup"
NS_ENV_BIN="${CONDA_PREFIX}/bin"
PYTHON_EXE="${NS_ENV_BIN}/python"
NS_TRAIN="${NS_ENV_BIN}/ns-train"
NS_EXPORT="${NS_ENV_BIN}/ns-export"

# 输出目录
OUTPUT_DIR="$DA3_DIR/output/nerfstudio_3dgs/$SCENE_NAME"

# 激活环境
CONDA_ENV="gs_linux_backup"
CONDA_BASE="/home/ltx/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0

# ================= 帮助 =================
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "使用 Nerfstudio 的 splatfacto 进行 3DGS 训练"
    echo ""
    echo "用法: ./train_3dgs_nerfstudio.sh <COLMAP_DIR> <SCENE_NAME> [ITERATIONS]"
    echo ""
    echo "参数:"
    echo "  <COLMAP_DIR>   COLMAP输出目录（默认: output/sugar_streaming1_colmap）"
    echo "  <SCENE_NAME>   场景名称（默认: sugar_streaming1）"
    echo "  [ITERATIONS]   训练迭代数（默认: 15000）"
    echo ""
    echo "说明:"
    echo "  使用 Nerfstudio 的 splatfacto 方法进行 3DGS 训练"
    echo "  支持内置 Web 查看器和实时训练监控"
    echo ""
    echo "输出:"
    echo "  PLY: $OUTPUT_DIR/export/*.ply"
    exit 0
fi

if [ ! -d "$COLMAP_DIR" ]; then
    echo "❌ 错误: COLMAP目录不存在: $COLMAP_DIR"
    exit 1
fi

echo "==================== Nerfstudio 3DGS 训练 ===================="
echo "COLMAP目录: $COLMAP_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "输出目录: $OUTPUT_DIR"
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

# ================= 步骤 2: 准备Nerfstudio数据目录 =================
echo ""
echo "==== [2/3] 准备训练数据 ===="

NS_DATA_DIR="$OUTPUT_DIR/data"
mkdir -p "$NS_DATA_DIR"
mkdir -p "$NS_DATA_DIR/images"

# 复制COLMAP数据 - Nerfstudio需要 colmap/sparse/0 结构
echo "  复制COLMAP sparse数据..."
mkdir -p "$NS_DATA_DIR/colmap"
cp -r "$COLMAP_DIR/sparse" "$NS_DATA_DIR/colmap/"
echo "  ✅ COLMAP sparse数据已复制到 colmap/sparse/0/"

# 处理图像
if [ -L "$COLMAP_DIR/images" ]; then
    # 如果是软链接，解析其目标
    LINK_TARGET="$(readlink -f "$COLMAP_DIR/images")"
    echo "  检测到图像软链接: $LINK_TARGET"
    cp -r "$LINK_TARGET"/* "$NS_DATA_DIR/images/"
elif [ -d "$COLMAP_DIR/images" ]; then
    echo "  复制图像..."
    cp -r "$COLMAP_DIR"/images/* "$NS_DATA_DIR/images/"
else
    echo "❌ 错误: 未找到图像目录"
    exit 1
fi

IMAGE_COUNT=$(ls -1 "$NS_DATA_DIR/images"/*.jpg "$NS_DATA_DIR/images"/*.png "$NS_DATA_DIR/images"/*.jpeg 2>/dev/null | wc -l)
echo "  ✅ 复制了 $IMAGE_COUNT 张图像"

# 检查并显示图片分辨率和相机参数
if [ $IMAGE_COUNT -gt 0 ]; then
    FIRST_IMG=$(ls "$NS_DATA_DIR/images"/*.jpg "$NS_DATA_DIR/images"/*.png "$NS_DATA_DIR/images"/*.jpeg 2>/dev/null | head -1)
    if [ -n "$FIRST_IMG" ]; then
        IMG_RES=$(python3 -c "from PIL import Image; img = Image.open('$FIRST_IMG'); print(f'{img.width} x {img.height}')" 2>/dev/null)
        if [ -n "$IMG_RES" ]; then
            echo "  📷 图片分辨率: $IMG_RES"
        fi
    fi

    # 读取COLMAP相机参数（如果存在文本格式）
    CAMERAS_TXT="$COLMAP_DIR/sparse/0/cameras.txt"
    if [ -f "$CAMERAS_TXT" ]; then
        CAM_INFO=$(grep -v '^#' "$CAMERAS_TXT" | head -1)
        if [ -n "$CAM_INFO" ]; then
            echo "  📐 COLMAP相机参数: $CAM_INFO"
        fi
    fi
fi

echo ""
echo "  🎞️  训练配置: $IMAGE_COUNT 张图像, $ITERATIONS 次迭代"
echo "  🎚️  高斯球限制: max=200万, stop-split-at=$((ITERATIONS - 3000))"
echo ""

# ================= 步骤 3: 使用Nerfstudio训练 =================
echo ""
echo "==== [3/3] 训练 Nerfstudio Splatfacto ($ITERATIONS 迭代) ===="

export SETUPTOOLS_USE_DISTUTILS=stdlib

echo "  开始训练..."
echo "  Web查看器将在训练启动后可用"
echo ""

cd "$DA3_DIR"

# 清理端口6006（nerfstudio默认端口）
PORT_PID=$(netstat -nlp 2>/dev/null | grep :6006 | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PORT_PID" ]; then
    echo "  清理端口 6006..."
    kill -9 $PORT_PID 2>/dev/null || true
    sleep 1
fi

# 使用 ns-train splatfacto
# 高斯椭球数量控制参数
$NS_TRAIN splatfacto \
    --data "$NS_DATA_DIR" \
    --output-dir "$OUTPUT_DIR/outputs" \
    --experiment-name "$SCENE_NAME" \
    --pipeline.model.random-init "False" \
    --max-num-iterations "$ITERATIONS" \
    --viewer.quit-on-train-completion "True" \
    --pipeline.model.densify-grad-thresh "0.0004" \
    --pipeline.model.cull-alpha-thresh "0.005" \
    --pipeline.model.stop-split-at "$((ITERATIONS - 3000))" \
    --pipeline.model.max-gs-num "2000000" \
    colmap \
    --orientation-method "none" \
    --center-method "poses" \
    --auto-scale-poses "True"

echo ""
echo "==================== ✨ 训练完成! ===================="
echo ""

# ================= 步骤 4: 导出PLY =================
echo "==== [4/4] 导出 PLY 文件 ===="

# 查找最新的 config.yml
CONFIG_PATH=$(find "$OUTPUT_DIR/outputs/$SCENE_NAME" -name "config.yml" -type f | head -1)

if [ -z "$CONFIG_PATH" ]; then
    echo "⚠️ 未找到 config.yml，无法导出PLY"
    echo "   请检查训练输出目录: $OUTPUT_DIR/outputs/$SCENE_NAME"
else
    echo "  找到配置: $CONFIG_PATH"

    mkdir -p "$OUTPUT_DIR/export"

    $NS_EXPORT gaussian-splat \
        --load-config "$CONFIG_PATH" \
        --output-dir "$OUTPUT_DIR/export"

    # 查找导出的PLY
    PLY_FILE=$(find "$OUTPUT_DIR/export" -name "*.ply" -type f | head -1)

    if [ -n "$PLY_FILE" ]; then
        PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
        echo ""
        echo "✅ 导出成功!"
        echo ""
        echo "📁 输出文件:"
        echo "  PLY: $PLY_FILE ($PLY_SIZE)"
        echo ""
        echo "查看方法:"
        echo "  1. SuperSplat (在线): https://playcanvas.com/supersplat/editor"
        echo "     拖拽 PLY 文件即可"
        echo ""
        echo "  2. 在训练过程中查看:"
        echo "     Nerfstudio 会启动 Web 查看器，默认端口 6006"
        echo "     在浏览器打开: http://localhost:6006"
    else
        echo "⚠️ 导出目录中未找到 PLY 文件"
    fi
fi

echo ""
echo "==================== 训练总结 ===================="
echo "训练输出: $OUTPUT_DIR/outputs/$SCENE_NAME"
echo "导出目录: $OUTPUT_DIR/export"
echo ""
