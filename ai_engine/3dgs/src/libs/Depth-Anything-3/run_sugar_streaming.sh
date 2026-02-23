#!/bin/bash
# SuGaR 视频处理脚本 - DA3-Streaming
# 使用 DA3-Streaming 处理超长视频，支持 3DGS 输出

set -e

# 配置
VIDEO_PATH="/home/ltx/projects/Depth-Anything-3/video3.mp4"
EXTRACT_DIR="/home/ltx/projects/Depth-Anything-3/output/sugar_streaming1/extracted"
OUTPUT_DIR="/home/ltx/projects/Depth-Anything-3/output/sugar_streaming1"
CONFIG="/home/ltx/projects/Depth-Anything-3/da3_streaming/configs/base_config.yaml"

# 抽帧参数
FPS=5  # 每秒抽 1 帧
SCALE="scale=720:-1"  # 720p（最长边 720 像素）

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup

echo "============================================================"
echo "DA3-Streaming: SuGaR 视频处理"
echo "============================================================"
echo "视频路径: $VIDEO_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "抽帧率: $FPS FPS"
echo "分辨率: 720p"
echo "============================================================"

# 创建目录
mkdir -p "$EXTRACT_DIR"
mkdir -p "$OUTPUT_DIR"

# 第一步：抽帧
echo ""
echo "[1/3] 从视频抽帧..."
ffmpeg -i "$VIDEO_PATH" \
  -vf "fps=$FPS,$SCALE" \
  "$EXTRACT_DIR/frame_%06d.png"

# 统计帧数
FRAME_COUNT=$(ls "$EXTRACT_DIR" | wc -l)
echo "✅ 抽帧完成：$FRAME_COUNT 帧"

# 第二步：运行 DA3-Streaming
echo ""
echo "[2/3] 运行 DA3-Streaming..."
echo "  模型：DA3-GIANT (1.15B 参数)"
echo "  处理方式：Streaming + 闭环检测"
echo "  输出：点云 + 深度图"
echo ""

HF_ENDPOINT=https://hf-mirror.com python da3_streaming/da3_streaming.py \
  --image_dir "$EXTRACT_DIR" \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ DA3-Streaming 完成！"

# 第三步：结果说明
echo ""
echo "[3/3] 输出文件："
echo ""
echo "主输出："
echo "  $OUTPUT_DIR/camera_poses.txt - 相机位姿"
echo "  $OUTPUT_DIR/intrinsic.txt - 相机内参"
echo "  $OUTPUT_DIR/pcd/combined_pcd.ply - 合并点云（3DGS 可用）"
echo ""
echo "详细输出（每帧）："
echo "  $OUTPUT_DIR/results_output/ - 每帧的 RGB、深度、置信度"
echo ""
echo "============================================================"
echo "处理完成！"
echo "============================================================"

# 提示
echo ""
echo "💡 点云查看："
echo "   MeshLab: meshlab $OUTPUT_DIR/pcd/combined_pcd.ply"
echo "   CloudCompare: cloudcompare $OUTPUT_DIR/pcd/combined_pcd.ply"
echo ""
echo "💡 深度视频生成："
echo "   cd $OUTPUT_DIR/results_output"
echo "   ffmpeg -framerate $FPS -i depth_%06d.png depth_video.mp4"
