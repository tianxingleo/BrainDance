#!/usr/bin/env python3
"""
DA3 3DGS 直接生成脚本

使用 DA3 的 Python API 直接处理图像并生成 3D Gaussians 渲染视频
无需 Gradio UI
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/ltx/projects/Depth-Anything-3/src')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import VideoHandler, ImagesHandler


def main():
    parser = argparse.ArgumentParser(description='DA3 3DGS Generator')
    parser.add_argument('--input', type=str, required=True,
                        help='输入：图像目录、视频文件或单张图像')
    parser.add_argument('--output-dir', type=str, default='./output/da3_3dgs',
                        help='输出目录')
    parser.add_argument('--model-dir', type=str, default='./weights',
                        help='模型目录')
    parser.add_argument('--gs-trj-mode', type=str, default='extend',
                        choices=['original', 'smooth', 'interpolate', 'wander', 'dolly_zoom', 'extend'],
                        help='3DGS 轨迹模式')
    parser.add_argument('--gs-quality', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='3DGS 视频质量')
    parser.add_argument('--export-format', type=str, default='mini_npz-gs_ply',
                        help='导出格式 (例如: mini_npz-gs_ply, mini_npz-gs_video, glb)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备: cuda 或 cpu')
    parser.add_argument('--process-res', type=int, default=504,
                        help='处理分辨率 (越小越省显存，建议: 392, 336, 280)')

    args = parser.parse_args()

    print("=" * 60)
    print("DA3 3DGS 生成工具")
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"轨迹模式: {args.gs_trj_mode}")
    print(f"视频质量: {args.gs_quality}")
    print("=" * 60)
    print()

    # 检查输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误：输入路径不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 DA3
    print("🔧 初始化 DA3 模型...")
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  警告: 指定了 cuda 但不可用，切换到 cpu")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"📡 使用设备: {device}")
    
    try:
        da3 = DepthAnything3.from_pretrained(args.model_dir).to(device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and device.type == 'cuda':
            print("❌ 初始化失败: 显存不足。尝试使用 --device cpu")
            sys.exit(1)
        raise e
        
    da3.eval()
    print("✅ 模型加载完成")
    print()

    # 处理输入
    try:
        if input_path.is_file():
            if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                print(f"🎬 处理视频: {input_path}")
                print()

                # 1. 提取视频帧
                image_paths = VideoHandler.process(str(input_path), str(output_dir), fps=1.0)

                # 2. 处理提取的帧
                print(f"🔮 模型推理及 3DGS 生成 (共 {len(image_paths)} 帧)...")
                result = da3.inference(
                    image=image_paths,
                    export_dir=str(output_dir),
                    process_res=args.process_res,
                    infer_gs=True,
                    export_format=args.export_format,
                    export_kwargs={
                        "gs_video": {
                            "trj_mode": args.gs_trj_mode,
                            "video_quality": args.gs_quality
                        }
                    }
                )

                print(f"✅ 视频处理完成")
                print(f"📁 输出目录: {output_dir}")
                print()

                # 检查是否有 gaussians (Prediction object has gaussians if infer_gs is True)
                if hasattr(result, 'gaussians') and result.gaussians is not None:
                    print("🎉 3D Gaussians (PLY) 生成成功！")
                    print(f"📄 查看 PLY 文件: {output_dir}/gs_ply/0000.ply")
                else:
                    print("ℹ️  当前模型不支持或未成功生成 3D Gaussians")
            else:
                # 单张图像
                print(f"🖼️  处理单张图像: {input_path}")
                result = da3.inference(
                    image=[str(input_path)],
                    export_dir=str(output_dir),
                    process_res=args.process_res,
                    infer_gs=True,
                    export_format='glb-gs_ply'
                )
                print(f"✅ 图像处理完成: {output_dir}")

        elif input_path.is_dir():
            print(f"📁 处理图像目录: {input_path}")
            image_paths = ImagesHandler.process(str(input_path))
            result = da3.inference(
                image=image_paths,
                export_dir=str(output_dir),
                process_res=args.process_res,
                infer_gs=True,
                export_format='glb-gs_ply',
                export_kwargs={
                    "gs_video": {
                        "trj_mode": args.gs_trj_mode,
                        "video_quality": args.gs_quality
                    }
                }
            )
            print(f"✅ 目录处理完成: {output_dir}")
            
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and device.type == 'cuda':
            print("\n" + "!"*40)
            print("❌ 报错：显存溢出 (Out of Memory)!")
            print("建议措施：")
            print(f"1. 减小分辨率: 运行命令时加上 --process-res 336 (当前为 {args.process_res})")
            print("2. 在内存运行: 运行命令时加上 --device cpu (速度会慢很多)")
            print("!"*40 + "\n")
            sys.exit(1)
        raise e

    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print()
    print("📂 输出文件位置:")
    print(f"   {output_dir}")
    print()

    # 列出输出文件
    if output_dir.exists():
        print("📄 生成的文件:")
        for f in sorted(output_dir.rglob('*')):
            if f.is_file():
                size = f.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"   - {f.relative_to(output_dir)} ({size_mb:.2f} MB)")

    print()
    print("💡 查看结果：")
    print(f"   glTF 模型: {output_dir}/scene.glb")


if __name__ == '__main__':
    main()
