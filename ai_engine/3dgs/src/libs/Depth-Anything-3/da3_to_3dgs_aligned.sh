#!/bin/bash
# 融合方案: Depth Anything 3 -> COLMAP 对齐 -> 3DGS 训练 -> Open3D 扶正
#
# 融合了方案 A (COLMAP model_aligner) 和方案 B (Open3D RANSAC) 的双重对齐 Pipeline
#
# 流程:
#   1. DA3 → COLMAP 格式转换
#   2. 🅰️ COLMAP model_aligner 平面对齐 (训练前，对齐相机+点云)
#   3. 整理数据目录
#   4. Vanilla 3DGS 训练
#   5. 🅱️ Open3D RANSAC 自动扶正 (训练后，精细校正输出PLY)
#
# 双重对齐的优势:
#   - COLMAP 对齐在训练前，相机位姿和点云一起旋转，训练本身受益于正确朝向
#   - Open3D 对齐在训练后，作为二次校正/安全网，确保最终 PLY 完全水平
#   - 如果 COLMAP 对齐失败，Open3D 仍可独立工作
#   - 如果 Open3D 检测到模型已经正确朝向，会跳过旋转
#
# 用法: ./da3_to_3dgs_aligned.sh <DA3输出目录> <场景名称> [选项...]
#
# 输出:
#   - 原始 PLY:  .../point_cloud.ply
#   - 扶正 PLY:  .../point_cloud_aligned.ply (如果 Open3D 进行了校正)

set -e

# ================= 默认配置 =================
DA3_OUTPUT_DIR=""
SCENE_NAME=""
ITERATIONS=30000
COLMAP_MAX_ERROR=0.02
OPEN3D_THRESHOLD=0.02
TRANSLATE_TO_GROUND=false
SKIP_COLMAP_ALIGN=false
SKIP_OPEN3D_ALIGN=false

# ================= 解析参数 =================
show_help() {
    echo "融合方案: DA3 → COLMAP 对齐 → 3DGS → Open3D 扶正"
    echo ""
    echo "用法: ./da3_to_3dgs_aligned.sh <DA3输出目录> <场景名称> [选项...]"
    echo ""
    echo "必填参数:"
    echo "  <DA3输出目录>           DA3的输出目录"
    echo "  <场景名称>              场景名称"
    echo ""
    echo "可选参数:"
    echo "  --iterations N          训练迭代数 (默认: 30000)"
    echo "  --colmap_error F        COLMAP对齐最大误差,米 (默认: 0.02)"
    echo "  --open3d_threshold F    Open3D RANSAC距离阈值,米 (默认: 0.02)"
    echo "  --translate_to_ground   将地面平移到 Z=0"
    echo "  --skip_colmap           跳过 COLMAP 对齐 (仅用方案B)"
    echo "  --skip_open3d           跳过 Open3D 扶正 (仅用方案A)"
    echo "  -h, --help              显示帮助"
    echo ""
    echo "模式示例:"
    echo "  # 双重对齐 (默认，推荐)"
    echo "  ./da3_to_3dgs_aligned.sh output/sugar_streaming my_scene"
    echo ""
    echo "  # 仅 COLMAP 对齐 (等效方案A)"
    echo "  ./da3_to_3dgs_aligned.sh output/sugar_streaming my_scene --skip_open3d"
    echo ""
    echo "  # 仅 Open3D 扶正 (等效方案B)"
    echo "  ./da3_to_3dgs_aligned.sh output/sugar_streaming my_scene --skip_colmap"
    echo ""
    echo "  # 自定义参数"
    echo "  ./da3_to_3dgs_aligned.sh output/sugar_streaming my_scene \\"
    echo "      --iterations 50000 --colmap_error 0.05 --open3d_threshold 0.03 \\"
    echo "      --translate_to_ground"
    echo ""
    echo "输出:"
    echo "  原始 PLY:  SuGaR/output/3dgs/<场景>/point_cloud/iteration_<N>/point_cloud.ply"
    echo "  扶正 PLY:  SuGaR/output/3dgs/<场景>/point_cloud/iteration_<N>/point_cloud_aligned.ply"
    echo ""
    echo "对比其他 Pipeline:"
    echo "  da3_to_3dgs.sh                   无对齐"
    echo "  da3_to_3dgs_aligned_colmap.sh    仅方案A (COLMAP)"
    echo "  da3_to_3dgs_aligned_open3d.sh    仅方案B (Open3D)"
    echo "  da3_to_3dgs_aligned.sh           融合双重对齐 (本脚本)"
    exit 0
}

# 解析位置参数和命名参数
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --colmap_error)
            COLMAP_MAX_ERROR="$2"
            shift 2
            ;;
        --open3d_threshold)
            OPEN3D_THRESHOLD="$2"
            shift 2
            ;;
        --translate_to_ground)
            TRANSLATE_TO_GROUND=true
            shift
            ;;
        --skip_colmap)
            SKIP_COLMAP_ALIGN=true
            shift
            ;;
        --skip_open3d)
            SKIP_OPEN3D_ALIGN=true
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# 恢复位置参数
DA3_OUTPUT_DIR="${POSITIONAL[0]:-output/sugar_streaming}"
SCENE_NAME="${POSITIONAL[1]:-my_scene}"

# ================= 路径配置 =================
DA3_DIR="/home/ltx/projects/Depth-Anything-3"
SUGAR_DIR="/home/ltx/projects/SuGaR"
COLMAP_TEXT_DIR="$DA3_OUTPUT_DIR/colmap_text"
COLMAP_ALIGNED_DIR="$DA3_OUTPUT_DIR/colmap_text/sparse/aligned"
SUGAR_DATA_DIR="$SUGAR_DIR/data/$SCENE_NAME"
GS_OUTPUT_DIR="$SUGAR_DIR/output/3dgs/$SCENE_NAME"
ALIGN_SCRIPT="$DA3_DIR/auto_align_ply.py"

# 激活环境
CONDA_ENV="gs_linux_backup"
CONDA_BASE="/home/ltx/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0

# ================= 验证 =================
if [ ! -d "$DA3_OUTPUT_DIR" ]; then
    echo "❌ 错误: DA3输出目录不存在: $DA3_OUTPUT_DIR"
    exit 1
fi

# 计算总步骤数
TOTAL_STEPS=3
STEP=0
if [ "$SKIP_COLMAP_ALIGN" = false ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [ "$SKIP_OPEN3D_ALIGN" = false ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi

# 确定对齐模式描述
if [ "$SKIP_COLMAP_ALIGN" = true ] && [ "$SKIP_OPEN3D_ALIGN" = true ]; then
    ALIGN_MODE="无对齐 (等效 da3_to_3dgs.sh)"
elif [ "$SKIP_COLMAP_ALIGN" = true ]; then
    ALIGN_MODE="仅 Open3D 扶正 (方案B)"
elif [ "$SKIP_OPEN3D_ALIGN" = true ]; then
    ALIGN_MODE="仅 COLMAP 对齐 (方案A)"
else
    ALIGN_MODE="双重对齐 (COLMAP + Open3D)"
fi

echo "==================== DA3 → 对齐 → 3DGS → 扶正 (融合方案) ===================="
echo "DA3输出目录: $DA3_OUTPUT_DIR"
echo "场景名称: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"
echo "对齐模式: $ALIGN_MODE"
if [ "$SKIP_COLMAP_ALIGN" = false ]; then
    echo "  COLMAP 误差阈值: $COLMAP_MAX_ERROR"
fi
if [ "$SKIP_OPEN3D_ALIGN" = false ]; then
    echo "  Open3D RANSAC 阈值: $OPEN3D_THRESHOLD"
    echo "  平移到地面: $TRANSLATE_TO_GROUND"
fi
echo "输出目录: $GS_OUTPUT_DIR"
echo ""

# 记录对齐结果
COLMAP_ALIGN_OK=false
OPEN3D_ALIGN_OK=false

# ================= 步骤 1: 转换为 COLMAP 格式 =================
STEP=$((STEP + 1))
echo "==== [$STEP/$TOTAL_STEPS] 转换 DA3 → COLMAP 格式 ===="

mkdir -p "$COLMAP_TEXT_DIR"

if [ -f "$COLMAP_TEXT_DIR/sparse/0/cameras.bin" ] && [ -f "$COLMAP_TEXT_DIR/sparse/0/images.bin" ]; then
    echo "  ✅ 已有COLMAP数据，跳过转换"
else
    python3 "$DA3_DIR/convert_da3_to_colmap.py" \
        --base_dir "$DA3_OUTPUT_DIR" \
        --output_dir "$COLMAP_TEXT_DIR"

    python3 "$DA3_DIR/colmap_text_to_binary.py" \
        "$COLMAP_TEXT_DIR/sparse/0"

    echo "  ✅ COLMAP 转换完成"
fi

# ================= 步骤 2 (可选): COLMAP 平面对齐 =================
if [ "$SKIP_COLMAP_ALIGN" = false ]; then
    STEP=$((STEP + 1))
    echo ""
    echo "==== [$STEP/$TOTAL_STEPS] 🅰️ COLMAP 平面对齐 (model_aligner) ===="

    mkdir -p "$COLMAP_ALIGNED_DIR"

    echo "  输入: $COLMAP_TEXT_DIR/sparse/0"
    echo "  输出: $COLMAP_ALIGNED_DIR"
    echo "  方式: plane (曼哈顿世界假设)"
    echo "  误差: $COLMAP_MAX_ERROR"
    echo ""

    if colmap model_aligner \
        --input_path "$COLMAP_TEXT_DIR/sparse/0" \
        --output_path "$COLMAP_ALIGNED_DIR" \
        --ref_is_gps 0 \
        --alignment_type plane \
        --alignment_max_error "$COLMAP_MAX_ERROR" 2>&1; then

        if [ -f "$COLMAP_ALIGNED_DIR/cameras.bin" ] && [ -f "$COLMAP_ALIGNED_DIR/images.bin" ]; then
            echo "  ✅ COLMAP 平面对齐成功"
            COLMAP_ALIGN_OK=true
        else
            echo "  ⚠️ COLMAP 对齐输出文件缺失，回退使用原始模型"
        fi
    else
        echo "  ⚠️ COLMAP model_aligner 执行失败，回退使用原始模型"
    fi
fi

# ================= 步骤 3: 整理数据目录 =================
STEP=$((STEP + 1))
echo ""
echo "==== [$STEP/$TOTAL_STEPS] 整理数据目录 ===="

if [ -d "$SUGAR_DATA_DIR" ]; then
    echo "  清理旧数据: $SUGAR_DATA_DIR"
    rm -rf "$SUGAR_DATA_DIR"
fi

mkdir -p "$SUGAR_DATA_DIR/sparse/0"
mkdir -p "$SUGAR_DATA_DIR/images"

# 复制 COLMAP 二进制 (优先使用对齐后的)
if [ "$COLMAP_ALIGN_OK" = true ]; then
    echo "  📐 使用 COLMAP 对齐后的模型"
    cp "$COLMAP_ALIGNED_DIR"/*.bin "$SUGAR_DATA_DIR/sparse/0/"
else
    if [ "$SKIP_COLMAP_ALIGN" = false ]; then
        echo "  ⚠️ COLMAP 对齐未成功，使用原始模型"
    fi
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
STEP=$((STEP + 1))
echo ""
echo "==== [$STEP/$TOTAL_STEPS] 训练 Vanilla 3DGS ($ITERATIONS 迭代) ===="

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

# ================= 步骤 5 (可选): Open3D 自动扶正 =================
PLY_FILE="$GS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"

if [ "$SKIP_OPEN3D_ALIGN" = false ]; then
    STEP=$((STEP + 1))
    echo ""
    echo "==== [$STEP/$TOTAL_STEPS] 🅱️ Open3D 自动扶正 (RANSAC 平面分割) ===="

    # 检查 Open3D
    if ! python3 -c "import open3d" 2>/dev/null; then
        echo "  ⚠️ Open3D 未安装，正在安装..."
        pip install open3d
        if [ $? -ne 0 ]; then
            echo "  ❌ Open3D 安装失败，跳过扶正步骤"
            SKIP_OPEN3D_ALIGN=true
        else
            echo "  ✅ Open3D 安装完成"
        fi
    fi

    if [ "$SKIP_OPEN3D_ALIGN" = false ]; then
        if [ ! -f "$PLY_FILE" ]; then
            echo "  ❌ 未找到 PLY 文件: $PLY_FILE"
            echo "  跳过扶正步骤"
        elif [ ! -f "$ALIGN_SCRIPT" ]; then
            echo "  ❌ 扶正脚本不存在: $ALIGN_SCRIPT"
            echo "  跳过扶正步骤"
        else
            PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
            echo "  输入 PLY: $PLY_FILE ($PLY_SIZE)"

            ALIGNED_PLY_FILE="$GS_OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/point_cloud_aligned.ply"

            # 构建参数
            ALIGN_ARGS="$PLY_FILE $ALIGNED_PLY_FILE --distance_threshold $OPEN3D_THRESHOLD"
            if [ "$TRANSLATE_TO_GROUND" = true ]; then
                ALIGN_ARGS="$ALIGN_ARGS --translate_to_ground"
            fi

            cd "$DA3_DIR"
            if python3 "$ALIGN_SCRIPT" $ALIGN_ARGS; then
                if [ -f "$ALIGNED_PLY_FILE" ]; then
                    ALIGNED_SIZE=$(du -h "$ALIGNED_PLY_FILE" | cut -f1)
                    echo "  ✅ Open3D 扶正完成 ($ALIGNED_SIZE)"
                    OPEN3D_ALIGN_OK=true
                fi
            else
                echo "  ⚠️ Open3D 扶正执行失败"
            fi
        fi
    fi
fi

# ================= 完成 =================
echo ""
echo "==================== ✨ Pipeline 完成! (融合方案) ===================="
echo ""

# 汇总对齐结果
echo "📊 对齐状态汇总:"
if [ "$SKIP_COLMAP_ALIGN" = true ]; then
    echo "  🅰️ COLMAP 对齐:  已跳过 (--skip_colmap)"
elif [ "$COLMAP_ALIGN_OK" = true ]; then
    echo "  🅰️ COLMAP 对齐:  ✅ 成功 (训练前已对齐相机+点云)"
else
    echo "  🅰️ COLMAP 对齐:  ❌ 失败 (使用了原始未对齐模型训练)"
fi

if [ "$SKIP_OPEN3D_ALIGN" = true ]; then
    echo "  🅱️ Open3D 扶正:  已跳过 (--skip_open3d)"
elif [ "$OPEN3D_ALIGN_OK" = true ]; then
    echo "  🅱️ Open3D 扶正:  ✅ 成功 (训练后精细校正)"
else
    echo "  🅱️ Open3D 扶正:  ❌ 失败"
fi
echo ""

# 输出文件信息
if [ -f "$PLY_FILE" ]; then
    PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
    echo "输出文件:"
    echo "  原始 PLY: $PLY_FILE ($PLY_SIZE)"

    if [ "$OPEN3D_ALIGN_OK" = true ] && [ -f "$ALIGNED_PLY_FILE" ]; then
        ALIGNED_SIZE=$(du -h "$ALIGNED_PLY_FILE" | cut -f1)
        echo "  扶正 PLY: $ALIGNED_PLY_FILE ($ALIGNED_SIZE)"
        echo ""
        echo "  💡 推荐使用扶正后的 PLY 文件"
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
    echo ""
    echo "  4. 单独对其他 PLY 文件扶正:"
    echo "     cd $DA3_DIR"
    echo "     python auto_align_ply.py <input.ply> <output.ply>"
else
    echo "⚠️ 未找到 PLY 文件，请检查训练日志"
    ls -la "$GS_OUTPUT_DIR/point_cloud/" 2>/dev/null || echo "  point_cloud 目录不存在"
fi
echo ""
