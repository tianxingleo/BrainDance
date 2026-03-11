# 功能：实现3DGS生成主流水线，协调各模块完成完整3D重建流程
# 实现：按顺序调用各个功能模块，处理从视频到3D模型的完整流程
# 逻辑：1. 视频抽帧与预处理 2. AI质检 3. 位姿解算 4. AI语义分割 5. 3DGS训练与导出
# 包含：run主函数、各模块实例化、流程控制逻辑、日志回调机制
import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# 引入基类
from src.core.pipeline_base import BasePipeline

# 引入配置
from src.config import PipelineConfig

# 引入所有业务模块
from src.modules.image_proc import ImageProcessor
from src.modules.glomap_runner import GlomapRunner
from src.modules.ai_segmentor import AISegmentor
from src.modules.nerf_engine import NerfstudioEngine
from src.modules.scene_analyzer import SceneAnalyzer
from src.modules.da3_runner import DA3Runner
from src.modules.spatial_anchor import SpatialAnchorExtractor

# 引入辅助工具
from src.utils.common import format_duration

class Video3DGSPipeline(BasePipeline):
    """
    【视频 -> 3DGS】标准流水线
    逻辑：视频抽帧 -> AI质检 -> GLOMAP解算 -> AI分割 -> 3DGS训练 -> 导出PLY
    """

    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        
        self.log(f"🎬 启动视频转 3DGS 流水线...")
        self.log(f"📄 输入文件: {input_path}")

        # ==========================================
        # 0. 初始化配置 (动态构建 Config)
        # ==========================================
        # 这里的 input_path 是 Worker 下载好的视频文件路径 (例如 /tmp/work/video.mp4)
        # 我们基于这个路径和 params 初始化 PipelineConfig
        
        video_path_obj = Path(input_path)
        
        # 直接在括号里传参初始化
        cfg = PipelineConfig(
            project_name=self.scene_id,  # 传入场景名
            video_path=video_path_obj,   # 传入视频路径
            mapper_type=params.get('mapper_type', os.getenv("MAPPER_TYPE", "glomap"))
        )
        
        # 单独设置工作目录 (因为 PipelineConfig 可能默认计算的是别的路径)
        cfg.project_dir = Path(self.work_dir)
        
        # params 是一个字典，例如： {'fast_mode': True, 'quality': 'low'}

        if params.get('fast_mode'): 
            # 1. params.get('key'): 安全获取值。
            #    如果字典里有 'fast_mode' 且值为 True，条件成立。
            #    如果字典里没有这个 key，它会返回 None (不会报错)，条件不成立。

            cfg.iterations = 7000 
            # 2. 修改配置 (cfg)。
            #    cfg 是全局配置对象，默认 iterations 可能设的是 30000 (标准质量)。
            #    这里直接把它改为 7000，意味着训练步数减少，速度变快，但质量会下降。

            self.log("🚀 已启用极速模式 (Fast Mode)")
            # 3. 记录日志。
            #    告诉用户：“我看到你的备注了，正在按极速模式处理。”

        if params.get('quality'):
            # 4. 检查是否有 'quality' 这个参数
            
            # 可以在这里根据 params['quality'] 调整 cfg.min_quality_score 等
            pass 
            # 5. pass 是占位符，表示“以后这里要写代码，暂时先空着”。
            #    未来你可以写：
            #    if params['quality'] == 'high':
            #        cfg.min_quality_score = 80 (严格筛选图片)

        global_start_time = time.time()
        
        # 初始化返回的元数据
        pipeline_metadata = {}

        # ==========================================
        # 1. 实例化所有业务模块
        # ==========================================
        img_processor = ImageProcessor(cfg, log_callback=self.log)
        scene_analyzer = SceneAnalyzer(cfg)
        
        # 🟢 根据参数或配置决定使用的是哪种解算引擎
        mapper_type = params.get('mapper_type', cfg.mapper_type)
        if mapper_type == 'da3':
            mapper_runner = DA3Runner(cfg, log_callback=self.log)
            self.log("    -> 使用引擎: Depth Anything 3")
        else:
            mapper_runner = GlomapRunner(cfg)
            self.log("    -> 使用引擎: GLOMAP")
            
        ai_segmentor = AISegmentor(cfg)
        nerf_engine = NerfstudioEngine(cfg)

        # ==========================================
        # Step 1: 数据准备 (视频抽帧)
        # ==========================================
        self.log(f"🎬 [1/4] 开始视频抽帧与图片预处理...")
        
        # 确保项目目录存在 (Worker 可能已经创建了，但保险起见)
        cfg.project_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份视频到项目目录 (可选，保持目录结构整洁)
        dest_video_path = cfg.project_dir / video_path_obj.name
        if not dest_video_path.exists():
            shutil.copy(str(video_path_obj), str(dest_video_path))
        
        # 抽帧临时目录
        temp_dir = cfg.project_dir / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"    -> 正在进行 FFmpeg 抽帧 (FPS=5, 最长边限制 1920px, Lanczos 超采样)...")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(dest_video_path),
                "-vf", "fps=5,scale=1920:1920:force_original_aspect_ratio=decrease:flags=lanczos",
                "-q:v", "2",
                "-map_metadata", "-1",  # 清除 EXIF，防止 COLMAP 读取原始视频 w/h 导致与实际帧尺寸不匹配
                str(temp_dir / "frame_%05d.jpg")
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg 抽帧失败: {e}")
            
        self.log(f"    -> FFmpeg 抽帧完成")

        # 图片清洗 (去模糊)
        img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)

        # 移动并限制图片数量
        raw_images_dir = cfg.project_dir / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)
        
        all_imgs = sorted(list(temp_dir.glob("*")))
        limit = cfg.max_images
        
        # 如果图片太多，进行均匀采样
        if len(all_imgs) > limit:
            indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
            all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
            
        for img in all_imgs:
            shutil.copy2(str(img), str(raw_images_dir / img.name))
            
        # 清理临时抽帧目录
        shutil.rmtree(temp_dir)
        self.log(f"    -> 图片准备完成，共 {len(all_imgs)} 张")

        # ==========================================
        # Step 1.5: AI 质检
        # ==========================================
        if cfg.enable_scene_analysis:
            self.log(f"🧐 [AI 质检] 阈值: {cfg.min_quality_score} 分")
            
            # 调用 SceneAnalyzer，注意传入 self.log 作为回调
            # 注意：这里需要适配 SceneAnalyzer 的 log_callback 签名
            # 假设 SceneAnalyzer 接受 log_callback(msg)
            passed, score, reason, tags, description, objects = scene_analyzer.run(
                raw_images_dir, 
                log_callback=lambda msg: self.log(msg) 
            )

            # 记录日志
            status_icon = "✅" if passed else "❌"
            self.log(f"    -> 结果: {status_icon} {score}分 (评价: {reason})")
            self.log(f"    -> 标签: {tags}")

            # 填充元数据
            pipeline_metadata.update({
                "ai_score": score,
                "ai_tags": tags,
                "ai_reason": reason,
                "ai_description": description,
                "ai_objects": objects
            })

            if not passed:
                err_msg = f"AI 质检不通过 ({score}分 < {cfg.min_quality_score}分): {reason}"
                self.log(err_msg, level="ERROR")
                raise RuntimeError(err_msg)

        # ==========================================
        # Step 2: 位姿解算
        # ==========================================
        self.log(f"⚙️ [2/4] 正在进行位姿解算 ({mapper_type.upper()})...")
        # 传递日志回调给 Runner (如果它支持的话)
        if not mapper_runner.run():
            err_msg = f"❌ Pipeline 中断：{mapper_type.upper()} 解算失败"
            self.log(err_msg, level="ERROR")
            raise RuntimeError(err_msg)
        self.log(f"    -> 位姿解算完成")

        # ==========================================
        # Step 3: AI 语义分割 (可选)
        # ==========================================
        # 可以在 params 里控制是否跳过
        if params.get('use_segmentation', True):
            self.log(f"🤖 [3/4] 正在进行 AI 语义分割...")
            ai_segmentor.run()
            self.log(f"    -> AI 处理完成")
        else:
            self.log(f"⏭️ [3/4] 跳过 AI 语义分割 (参数控制)")

        # ==========================================
        # Step 4: 3DGS 训练与导出
        # ==========================================
        self.log(f"🧠 [4/4] 开始 3DGS 训练 (迭代次数: {cfg.iterations})...")
        try:
            # 开始训练
            nerf_engine.train()
            self.log(f"    -> 训练完成，开始导出...")
            
            # 导出 PLY
            final_ply_path = nerf_engine.export()
            
            self.log(f"💾 导出 PLY 完成: {final_ply_path}")

            # ==========================================
            # Step 5: 空间语义锚点提取
            # ==========================================
            supabase_client = self.context.get('supabase')
            if supabase_client:
                anchor_extractor = SpatialAnchorExtractor(cfg, supabase_client)
                anchor_extractor.extract_and_save(
                    self.scene_id,
                    user_id=self.context.get("user_id"),
                    log_callback=self.log
                )
            else:
                self.log("⚠️ 未找到 Supabase 客户端，跳过空间语义锚点提取")

            # ==========================================
            # Step 6: 智能挑选最佳封图与初始视点
            # ==========================================
            import json
            webgl_poses_path = cfg.project_dir / "webgl_poses.json"
            if webgl_poses_path.exists():
                try:
                    with open(webgl_poses_path, "r") as f:
                        poses_data = json.load(f)
                    frames = poses_data.get("frames", [])
                    if frames:
                        # 使用 SceneAnalyzer 挑选最佳帧
                        best_idx, preview_reason = scene_analyzer.select_best_preview(
                            frames=frames, 
                            images_dir=str(cfg.project_dir / "raw_images"), 
                            log_callback=self.log
                        )
                        
                        best_frame = frames[best_idx]
                        pipeline_metadata["initial_camera_pose"] = best_frame.get("matrix")
                        pipeline_metadata["preview_selection_reason"] = preview_reason
                        
                        # 解析出对应的图片文件名
                        best_img_name = best_frame.get("id") or best_frame.get("image_url")
                        if best_img_name:
                            if best_img_name.startswith("images/"):
                                best_img_name = best_img_name[7:]
                            elif best_img_name.startswith("images\\"):
                                best_img_name = best_img_name[7:]
                                
                            preview_img = cfg.project_dir / "raw_images" / best_img_name
                            if not preview_img.exists():
                                preview_img = cfg.data_dir / "images" / best_img_name
                                
                            if preview_img.exists():
                                pipeline_metadata["preview_img_path"] = str(preview_img)
                                self.log(f"    -> 已提取初始视角和预览图: {best_img_name}")
                                self.log(f"    -> 封面选择理由: {preview_reason}")
                except Exception as e:
                    self.log(f"⚠️ 提取预览特征失败: {e}")

            # 上传 PLY 并在 model_assets 中写入记录（非强制，内部容错）
            try:
                self.upload_and_record(str(final_ply_path), pipeline_metadata, params)
            except Exception:
                # upload_and_record 内部已捕获异常，这里保证不抛出
                pass

            self.log(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")

            # 返回最终结果
            return str(final_ply_path), pipeline_metadata
            
        except Exception as e:
            self.log(f"❌ 训练/导出阶段失败: {e}", level="ERROR")
            raise e

    def cleanup(self):
        """
        重写清理逻辑：删除项目目录中的临时文件，保留 output
        """
        self.log("🧹正在清理临时文件...")
        # 可以在这里删除 raw_images, colmap 目录等，只保留 output
        # 暂时留空，或者根据 Config 决定是否删除
        pass
