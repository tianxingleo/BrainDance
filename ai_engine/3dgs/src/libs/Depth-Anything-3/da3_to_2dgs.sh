#!/bin/bash
# Depth Anything 3 -> 2D Gaussian Splatting 训练 Pipeline
#
# 2DGS 相比 3DGS 的优势:
#   - 使用 2D 高斯盘（而非 3D 椭球），天然更贴合表面
#   - 内置法线一致性约束（lambda_normal）和分布正则化（lambda_dist）
#   - 自带 TSDF mesh 导出（无需 SuGaR 的复杂 SDF 流程）
#   - 训练时间与 3DGS 相当，但几何质量更好
#
# 用法:
#   ./da3_to_2dgs.sh <DA3输出目录> <场景名称> [迭代次数] [导出mesh]
#
# 示例:
#   ./da3_to_2dgs.sh output/sugar_streaming my_scene           # 标准训练 30k
#   ./da3_to_2dgs.sh output/sugar_streaming my_scene 7000      # 快速预览 7k
#   ./da3_to_2dgs.sh output/sugar_streaming my_scene 30000 yes # 训练+导出mesh

set -e

# ================= 配置 =================
DA3_OUTPUT_DIR="${1:-output/sugar_streaming}"
SCENE_NAME="${2:-my_scene}"
ITERATIONS="${3:-30000}"
EXPORT_MESH="${4:-no}"  # yes/no

# 路径配置
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
DGS_DIR="/home/ltx/projects/2d-gaussian-splatting"
COLMAP_TEXT_DIR="$DA3_OUTPUT_DIR/colmap_text"
DGS_DATA_DIR="$DGS_DIR/data/$SCENE_NAME"
DGS_OUTPUT_DIR="$DGS_DIR/output/$SCENE_NAME"

# 激活环境
CONDA_ENV="gs_linux_backup"
CONDA_BASE="/home/ltx/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0
export QT_QPA_PLATFORM=offscreen

# WSL CUDA 路径修复
if [ -d "/usr/lib/wsl/lib" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# ================= 帮助 =================
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Depth Anything 3 -> 2D Gaussian Splatting Pipeline"
    echo ""
    echo "用法: ./da3_to_2dgs.sh <DA3输出目录> <场景名称> [迭代次数] [导出mesh]"
    echo ""
    echo "参数:"
    echo "  <DA3输出目录>   DA3的输出目录（默认: output/sugar_streaming）"
    echo "  <场景名称>      场景名称（默认: my_scene）"
    echo "  [迭代次数]      训练迭代数（默认: 30000）"
    echo "  [导出mesh]      是否导出mesh: yes/no（默认: no）"
    echo ""
    echo "迭代次数参考:"
    echo "  7000   快速预览（~5分钟）"
    echo "  15000  标准质量（~15分钟）"
    echo "  30000  高质量（~30分钟，推荐）"
    echo ""
    echo "2DGS vs 3DGS 对比:"
    echo "  3DGS: 3D椭球高斯，纯RGB重建，无几何约束"
    echo "  2DGS: 2D高斯盘，内置法线一致性+分布正则，几何质量更好"
    echo "        自带TSDF mesh导出（比SuGaR的SDF流程快10倍以上）"
    echo ""
    echo "示例:"
    echo "  ./da3_to_2dgs.sh output/sugar_streaming my_scene           # 纯2DGS"
    echo "  ./da3_to_2dgs.sh output/sugar_streaming my_scene 30000 yes # 2DGS + mesh"
    exit 0
fi

if [ ! -d "$DA3_OUTPUT_DIR" ]; then
    echo "❌ 错误: DA3输出目录不存在: $DA3_OUTPUT_DIR"
    exit 1
fi

echo "==================== DA3 → 2DGS Pipeline ===================="
echo "DA3输出目录: $DA3_OUTPUT_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "导出mesh: $EXPORT_MESH"
echo "输出目录: $DGS_OUTPUT_DIR"
echo ""
echo "2DGS 内置正则化:"
echo "  - lambda_normal = 0.05 (法线一致性，iteration > 7000 启用)"
echo "  - lambda_dist   = 0.0  (分布正则化，iteration > 3000 启用)"
echo "  ※ 无需额外SDF正则化！2D高斯盘天然贴合表面"
echo ""

# ================= 步骤 1: 转换为 COLMAP 格式 =================
echo "==== [1/4] 转换 DA3 → COLMAP 格式 ===="

mkdir -p "$COLMAP_TEXT_DIR"

# 检查是否已有转换结果
if [ -f "$COLMAP_TEXT_DIR/sparse/0/cameras.bin" ] && [ -f "$COLMAP_TEXT_DIR/sparse/0/images.bin" ]; then
    echo "  ✅ 已有COLMAP二进制数据，跳过转换"
else
    # 检查是否只有文本格式
    if [ -f "$COLMAP_TEXT_DIR/sparse/0/cameras.txt" ] && [ -f "$COLMAP_TEXT_DIR/sparse/0/images.txt" ]; then
        echo "  已有COLMAP文本数据，只需生成二进制..."
    else
        # 文本格式
        echo "  生成COLMAP文本格式..."
        python3 "$DA3_DIR/convert_da3_to_colmap.py" \
            --base_dir "$DA3_OUTPUT_DIR" \
            --output_dir "$COLMAP_TEXT_DIR"
    fi

    # 二进制格式
    echo "  生成COLMAP二进制格式..."
    python3 "$DA3_DIR/colmap_text_to_binary.py" \
        "$COLMAP_TEXT_DIR/sparse/0"

    echo "  ✅ COLMAP 转换完成"
fi

# ================= 步骤 2: 整理数据目录 =================
echo ""
echo "==== [2/4] 整理 2DGS 数据目录 ===="

if [ -d "$DGS_DATA_DIR" ]; then
    echo "  清理旧数据: $DGS_DATA_DIR"
    rm -rf "$DGS_DATA_DIR"
fi

mkdir -p "$DGS_DATA_DIR/sparse/0"
mkdir -p "$DGS_DATA_DIR/images"

# 复制 COLMAP 二进制
cp "$COLMAP_TEXT_DIR/sparse/0"/*.bin "$DGS_DATA_DIR/sparse/0/"

# 复制图像
LINK_TARGET="$(readlink -f "$COLMAP_TEXT_DIR/images" 2>/dev/null)"
if [ -z "$LINK_TARGET" ] || [ ! -d "$LINK_TARGET" ]; then
    LINK_TARGET="$DA3_OUTPUT_DIR/extracted"
fi
cp -r "$LINK_TARGET"/* "$DGS_DATA_DIR/images/"

IMAGE_COUNT=$(ls -1 "$DGS_DATA_DIR/images"/*.jpg "$DGS_DATA_DIR/images"/*.png "$DGS_DATA_DIR/images"/*.jpeg 2>/dev/null | wc -l)
echo "  ✅ 复制了 $IMAGE_COUNT 张图像到 $DGS_DATA_DIR/images/"
echo "  ✅ COLMAP 数据已复制到 $DGS_DATA_DIR/sparse/0/"

# ================= 步骤 3: 2DGS 训练 =================
echo ""
echo "==== [3/4] 训练 2D Gaussian Splatting ($ITERATIONS 迭代) ===="

# 清理端口
PORT_PID=$(netstat -nlp 2>/dev/null | grep :6009 | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PORT_PID" ]; then
    echo "  清理端口 6009..."
    kill -9 $PORT_PID 2>/dev/null || true
    sleep 1
fi

cd "$DGS_DIR"

echo "  开始训练..."
echo ""

python train.py \
    -s "$DGS_DATA_DIR" \
    -m "$DGS_OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations $ITERATIONS \
    --test_iterations $ITERATIONS

echo ""
echo "  ✅ 2DGS 训练完成！"

# ================= 步骤 4: 渲染 + Mesh导出（可选）=================
echo ""
echo "==== [4/4] 渲染结果 ===="

if [ "$EXPORT_MESH" = "yes" ] || [ "$EXPORT_MESH" = "true" ]; then
    echo "  渲染 + 导出 mesh..."
    python render.py \
        -m "$DGS_OUTPUT_DIR" \
        --iteration "$ITERATIONS" \
        --skip_test
    echo ""
    echo "  ✅ Mesh 导出完成！"
else
    echo "  渲染训练视角..."
    python render.py \
        -m "$DGS_OUTPUT_DIR" \
        --iteration "$ITERATIONS" \
        --skip_test \
        --skip_mesh
    echo ""
    echo "  ✅ 渲染完成（跳过mesh导出）"
fi

# ================= 完成 =================
echo ""
echo "==================== ✨ Pipeline 完成! ===================="
echo ""

# 查找输出文件
PLY_FILE="$DGS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
TRAIN_DIR="$DGS_OUTPUT_DIR/train/ours_$ITERATIONS"

echo "输出文件:"
if [ -f "$PLY_FILE" ]; then
    PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
    echo "  📄 2DGS PLY: $PLY_FILE ($PLY_SIZE)"
fi

if [ "$EXPORT_MESH" = "yes" ] || [ "$EXPORT_MESH" = "true" ]; then
    MESH_FILE="$TRAIN_DIR/fuse_post.ply"
    if [ -f "$MESH_FILE" ]; then
        MESH_SIZE=$(du -h "$MESH_FILE" | cut -f1)
        echo "  📐 Mesh:     $MESH_FILE ($MESH_SIZE)"
    fi
fi

RENDER_DIR="$TRAIN_DIR"
if [ -d "$RENDER_DIR" ]; then
    RENDER_COUNT=$(ls -1 "$RENDER_DIR"/*.png 2>/dev/null | wc -l)
    echo "  🖼️  渲染图:   $RENDER_DIR/ ($RENDER_COUNT 张)"
fi

echo ""
echo "查看方法:"
echo ""
echo "1. SuperSplat (在线 2DGS 查看器):"
echo "   https://playcanvas.com/supersplat/editor"
echo "   拖拽: $PLY_FILE"
echo ""
echo "2. 本地 2DGS 查看器:"
echo "   cd $DGS_DIR"
echo "   python view.py -s $DGS_DATA_DIR -m $DGS_OUTPUT_DIR"
echo ""

if [ "$EXPORT_MESH" = "yes" ] || [ "$EXPORT_MESH" = "true" ]; then
    echo "3. 查看 Mesh:"
    echo "   MeshLab: meshlab $TRAIN_DIR/fuse_post.ply"
    echo "   Blender: 导入 fuse_post.ply"
    echo ""
fi

echo "对比其他方案:"
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │ 方案                        │ 耗时      │ 几何约束  │ Mesh   │"
echo "  ├─────────────────────────────┼───────────┼──────────┼────────┤"
echo "  │ da3_to_2dgs.sh (本脚本)     │ ~30分钟   │ 法线一致 │ TSDF   │"
echo "  │ da3_to_3dgs.sh (纯3DGS)     │ ~20分钟   │ 无       │ 无     │"
echo "  │ da3_to_sugar_pipeline.sh    │ ~2-3小时  │ SDF+DN   │ SuGaR  │"
echo "  └─────────────────────────────┴───────────┴──────────┴────────┘"
echo ""
