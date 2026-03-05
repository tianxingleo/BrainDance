#!/bin/bash
# 方案 B: Depth Anything 3 -> 纯 3DGS 训练 -> Open3D 点云自动扶正
#
# 基于 da3_to_3dgs.sh，增加了 Open3D RANSAC 平面分割 + 自动扶正步骤
# 在训练完成后，对输出的 PLY 文件进行后处理自动扶正
#
# 用法: ./da3_to_3dgs_aligned_open3d.sh <DA3输出目录> <场景名称> [迭代次数] [距离阈值] [--translate_to_ground]
#
# 对比其他方案:
#   - da3_to_3dgs.sh:                 无对齐
#   - da3_to_3dgs_aligned_colmap.sh:  COLMAP 原生对齐 (训练前)
#   - 本脚本 (方案B):                 Open3D 后处理对齐 (训练后)
#
# 优点: 可控性高，RANSAC 参数可调，对任何 PLY 文件通用
# 缺点: 需要安装 Open3D
#
# 依赖: pip install open3d

set -e

# ================= 配置 =================
DA3_OUTPUT_DIR="${1:-output/sugar_streaming}"
SCENE_NAME="${2:-my_scene}"
ITERATIONS="${3:-30000}"            # 默认30k迭代（标准3DGS设置）
DISTANCE_THRESHOLD="${4:-0.02}"     # RANSAC 距离阈值 (米)
TRANSLATE_TO_GROUND="${5:-false}"   # 是否平移地面到 Z=0

# 路径配置
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
SUGAR_DIR="/home/ltx/projects/SuGaR"
COLMAP_TEXT_DIR="$DA3_OUTPUT_DIR/colmap_text"
SUGAR_DATA_DIR="$SUGAR_DIR/data/$SCENE_NAME"
GS_OUTPUT_DIR="$SUGAR_DIR/output/3dgs/$SCENE_NAME"

# Open3D 自动扶正脚本
ALIGN_SCRIPT="$DA3_DIR/auto_align_ply.py"

# 激活环境
CONDA_ENV="gs_linux_backup"
CONDA_BASE="/home/ltx/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0

# ================= 帮助 =================
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "方案 B: Depth Anything 3 -> 纯 3DGS -> Open3D 自动扶正"
    echo ""
    echo "用法: ./da3_to_3dgs_aligned_open3d.sh <DA3输出目录> <场景名称> [迭代次数] [距离阈值] [--translate_to_ground]"
    echo ""
    echo "参数:"
    echo "  <DA3输出目录>     DA3的输出目录（默认: output/sugar_streaming）"
    echo "  <场景名称>        场景名称（默认: my_scene）"
    echo "  [迭代次数]        训练迭代数（默认: 30000）"
    echo "  [距离阈值]        RANSAC距离阈值,米（默认: 0.02）"
    echo "  [--translate_to_ground]  平移地面到Z=0（默认: false）"
    echo ""
    echo "原理:"
    echo "  1. 正常训练 3DGS"
    echo "  2. 训练完成后，用 Open3D RANSAC 检测 PLY 中的地面"
    echo "  3. 计算旋转矩阵，将地面法向量对齐到 Z 轴"
    echo "  4. 保存扶正后的 PLY"
    echo ""
    echo "依赖:"
    echo "  pip install open3d"
    echo ""
    echo "输出:"
    echo "  原始 PLY:  .../point_cloud.ply"
    echo "  扶正 PLY:  .../point_cloud_aligned.ply"
    exit 0
fi

if [ ! -d "$DA3_OUTPUT_DIR" ]; then
    echo "❌ 错误: DA3输出目录不存在: $DA3_OUTPUT_DIR"
    exit 1
fi

# ================= 检查 Open3D =================
echo "==== 检查依赖 ===="
python3 -c "import open3d" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠️ Open3D 未安装，正在安装..."
    pip install open3d
    echo "  ✅ Open3D 安装完成"
else
    echo "  ✅ Open3D 已安装"
fi

if [ ! -f "$ALIGN_SCRIPT" ]; then
    echo "❌ 错误: 扶正脚本不存在: $ALIGN_SCRIPT"
    echo "  请确保 auto_align_ply.py 在 $DA3_DIR 目录下"
    exit 1
fi

echo ""
echo "==================== DA3 → 纯 3DGS → Open3D 扶正 (方案B) ===================="
echo "DA3输出目录: $DA3_OUTPUT_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "RANSAC距离阈值: $DISTANCE_THRESHOLD"
echo "平移到地面: $TRANSLATE_TO_GROUND"
echo "输出目录: $GS_OUTPUT_DIR"
echo ""

# ================= 步骤 1: 转换为 COLMAP 格式 =================
echo "==== [1/4] 转换 DA3 → COLMAP 格式 ===="

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
echo "==== [2/4] 整理数据目录 ===="

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
echo "==== [3/4] 训练 Vanilla 3DGS ($ITERATIONS 迭代) ===="

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
echo "  ✅ 3DGS 训练完成"

# ================= 步骤 4: Open3D 自动扶正 =================
echo ""
echo "==== [4/4] Open3D 自动扶正 (RANSAC 平面分割) ===="

PLY_FILE="$GS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"

if [ ! -f "$PLY_FILE" ]; then
    echo "❌ 错误: 未找到 PLY 文件: $PLY_FILE"
    echo "  训练可能失败，请检查日志"
    ls -la "$GS_OUTPUT_DIR/point_cloud/" 2>/dev/null || echo "  point_cloud 目录不存在"
    exit 1
fi

PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
echo "  输入 PLY: $PLY_FILE ($PLY_SIZE)"

# 生成输出路径
ALIGNED_PLY_FILE="$GS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud_aligned.ply"

# 构建命令参数
ALIGN_ARGS="$PLY_FILE $ALIGNED_PLY_FILE --distance_threshold $DISTANCE_THRESHOLD"
if [ "$TRANSLATE_TO_GROUND" = "true" ] || [ "$TRANSLATE_TO_GROUND" = "--translate_to_ground" ]; then
    ALIGN_ARGS="$ALIGN_ARGS --translate_to_ground"
fi

# 运行自动扶正
cd "$DA3_DIR"
python3 "$ALIGN_SCRIPT" $ALIGN_ARGS

if [ $? -eq 0 ] && [ -f "$ALIGNED_PLY_FILE" ]; then
    ALIGNED_SIZE=$(du -h "$ALIGNED_PLY_FILE" | cut -f1)
    echo ""
    echo "==================== ✨ 训练 + 扶正完成! (方案B: Open3D) ===================="
    echo ""
    echo "输出文件:"
    echo "  原始 PLY:  $PLY_FILE ($PLY_SIZE)"
    echo "  扶正 PLY:  $ALIGNED_PLY_FILE ($ALIGNED_SIZE)"
    echo ""
    echo "📐 扶正后的模型地面已对齐到 X-Y 平面"
    echo ""
    echo "查看方法:"
    echo "  1. SuperSplat (在线): https://playcanvas.com/supersplat/editor"
    echo "     拖拽 扶正PLY 文件即可"
    echo ""
    echo "  2. 本地渲染 (使用原始未扶正模型):"
    echo "     cd $SUGAR_DIR"
    echo "     python gaussian_splatting/render.py -m $GS_OUTPUT_DIR --iteration $ITERATIONS"
    echo ""
    echo "  3. 单独对其他 PLY 文件扶正:"
    echo "     python auto_align_ply.py <input.ply> <output.ply>"
    echo ""
    echo "  4. SIBR 查看器:"
    echo "     cd $SUGAR_DIR/gaussian_splatting/SIBR_viewers/install/bin"
    echo "     ./SIBR_gaussianViewer_app -m $GS_OUTPUT_DIR"
else
    echo ""
    echo "==================== ✨ 训练完成! (扶正失败) ===================="
    echo ""
    echo "⚠️ Open3D 自动扶正失败，但训练已成功完成"
    echo ""
    echo "输出文件:"
    echo "  PLY: $PLY_FILE ($PLY_SIZE)"
    echo ""
    echo "可手动扶正:"
    echo "  python auto_align_ply.py $PLY_FILE $ALIGNED_PLY_FILE"
fi
echo ""
