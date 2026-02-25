# src/modules/spatial_anchor.py
# 功能：实现空间语义锚点提取，将3DGS位姿与多模态语义结合存入向量数据库
# 实现：读取位姿数据，转换为WebGL坐标系，抽样关键帧，调用Qwen-VL打标，生成Embedding并存入Supabase
# 逻辑：1. 读取transforms 2. 坐标系转换 3. 抽样打标 4. 向量化 5. 存入数据库
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import PipelineConfig
from src.modules.scene_analyzer import SceneAnalyzer
from src.modules.rag_memory import RagMemory

class SpatialAnchorExtractor:
    def __init__(self, cfg: PipelineConfig, supabase_client):
        self.cfg = cfg
        self.supabase = supabase_client
        self.scene_analyzer = SceneAnalyzer(cfg)
        self.rag_memory = RagMemory(supabase_client)

    def extract_and_save(self, scene_id: str, log_callback=None):
        """执行空间语义锚点提取并保存到数据库"""
        if log_callback:
            log_callback("📍 [5/5] 开始提取空间语义锚点...")

        # 1. 查找 transforms.json 和 dataparser_transforms.json
        transforms_path = self.cfg.transforms_file
        if not transforms_path.exists():
            if log_callback: log_callback(f"⚠️ 未找到 {transforms_path}，跳过锚点提取")
            return False

        output_dir = self.cfg.project_dir / "outputs"
        search_path = output_dir / self.cfg.project_name / "splatfacto"
        try:
            run_dirs = sorted(list(search_path.glob("*")))
            dataparser_path = run_dirs[-1] / "dataparser_transforms.json"
        except IndexError:
            if log_callback: log_callback("⚠️ 未找到训练结果目录，跳过锚点提取")
            return False

        if not dataparser_path.exists():
            if log_callback: log_callback(f"⚠️ 未找到 {dataparser_path}，跳过锚点提取")
            return False

        # 2. 计算 WebGL 对齐的位姿
        webgl_poses = self._calculate_webgl_poses(transforms_path, dataparser_path)
        if not webgl_poses:
            if log_callback: log_callback("⚠️ 计算 WebGL 位姿失败，跳过锚点提取")
            return False

        # 保存 webgl_poses.json 到输出目录
        webgl_poses_path = self.cfg.project_dir / "webgl_poses.json"
        try:
            # 提取元数据 (FOV等)
            with open(transforms_path, 'r') as f:
                orig_data = json.load(f)
            fl_x = orig_data.get("fl_x")
            fl_y = orig_data.get("fl_y")
            w = orig_data.get("w")
            h = orig_data.get("h")
            camera_model = orig_data.get("camera_model")

            output_data = {
                "w": w,
                "h": h,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "camera_model": camera_model,
                "frames": webgl_poses
            }
            with open(webgl_poses_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            if log_callback: log_callback(f"    -> 已保存 WebGL 位姿到 {webgl_poses_path}")
        except Exception as e:
            if log_callback: log_callback(f"⚠️ 保存 webgl_poses.json 失败: {e}")

        # 3. 获取 model_id 和 user_id
        model_id = self._get_model_id(scene_id)
        if not model_id:
            if log_callback: log_callback(f"⚠️ 未在数据库中找到 scene_id={scene_id} 的模型，跳过锚点提取")
            return False
            
        bucket = os.getenv("SUPABASE_BUCKET", "braindance-assets")
        user_id = self.cfg.project_name.split('_')[0] if '_' in self.cfg.project_name else 'default_user'
        try:
            response = self.supabase.table("model_assets").select("user_id").eq("id", model_id).execute()
            if response.data and len(response.data) > 0:
                user_id = response.data[0]['user_id']
        except Exception:
            pass

        # 4. 抽样并打标
        sample_count = 10
        sampled_frames = random.sample(webgl_poses, min(sample_count, len(webgl_poses)))
        
        success_count = 0
        for frame in sampled_frames:
            image_name = frame['id']
            
            # 尝试从 data/images 找图片
            img_path = self.cfg.data_dir / "images" / image_name
            if not img_path.exists():
                # 尝试从 raw_images 找图片
                img_path = self.cfg.project_dir / "raw_images" / image_name
                
            if not img_path.exists():
                if log_callback: log_callback(f"    -> ⚠️ 找不到图片 {image_name}，跳过")
                continue
                
            # 🟢 [新增] 上传图片到 Supabase Storage
            try:
                remote_img_path = f"{user_id}/{scene_id}/output/images/{image_name}"
                with open(img_path, "rb") as f:
                    self.supabase.storage.from_(bucket).upload(
                        path=remote_img_path,
                        file=f,
                        file_options={"content-type": "image/jpeg", "x-upsert": "true", "upsert": "true"}
                    )
                if log_callback: log_callback(f"    -> ⬆️ 已上传图片 {image_name}")
            except Exception as e:
                if log_callback: log_callback(f"    -> ⚠️ 上传图片 {image_name} 失败: {e}")
                
            if log_callback: log_callback(f"    -> 正在分析视角 {image_name} ...")
            
            # 调用 Qwen-VL 打标
            tag = self._get_image_tag(img_path)
            if not tag:
                continue
                
            if log_callback: log_callback(f"    -> 识别结果: {tag}")
            
            # 5. 向量化并入库
            try:
                embedding = self.rag_memory.embed_text(tag)
                
                row = {
                    "model_id": model_id,
                    "image_name": image_name,
                    "transform_matrix": frame['matrix'],
                    "tag": tag,
                    "embedding": embedding # 阿里云 text-embedding-v2 维度为 1536
                }
                
                self.supabase.table("memory_poses").insert(row).execute()
                success_count += 1
            except Exception as e:
                if log_callback: log_callback(f"⚠️ 保存锚点失败: {e}")

        # 6. 上传 webgl_poses.json 到 Supabase Storage
        try:
            # 更新 webgl_poses.json 中的 image_url 为真实的云端路径
            # 只有被抽样并成功上传的图片才有真实的云端路径，其他的保持原样或置空
            # 为了简单起见，我们可以将所有图片的 image_url 都指向云端路径，
            # 即使有些图片没有被上传（前端可以处理 404 或者我们也可以选择上传所有图片）
            # 这里我们选择更新所有图片的 image_url
            for frame in webgl_poses:
                frame['image_url'] = f"{user_id}/{scene_id}/output/images/{frame['id']}"
                
            # 重新保存 webgl_poses.json
            with open(webgl_poses_path, 'w') as f:
                json.dump(output_data, f, indent=4)

            remote_path = f"{user_id}/{scene_id}/output/webgl_poses.json"
            with open(webgl_poses_path, "rb") as f:
                self.supabase.storage.from_(bucket).upload(
                    path=remote_path,
                    file=f,
                    file_options={"x-upsert": "true", "upsert": "true"}
                )
            if log_callback: log_callback(f"    -> ✅ 已上传 webgl_poses.json 到云端")
        except Exception as e:
            if log_callback: log_callback(f"⚠️ 上传 webgl_poses.json 失败: {e}")

        if log_callback:
            log_callback(f"✅ 空间语义锚点提取完成，成功保存 {success_count} 个锚点")
        return True

    def _calculate_webgl_poses(self, transforms_path: Path, dataparser_path: Path):
        """计算 WebGL 对齐的位姿 (基于 ns-export cameras 逻辑)"""
        try:
            # 1. 提取元数据 (FOV等)
            with open(transforms_path, 'r') as f:
                orig_data = json.load(f)
            fl_x = orig_data.get("fl_x")
            fl_y = orig_data.get("fl_y")
            w = orig_data.get("w")
            h = orig_data.get("h")
            camera_model = orig_data.get("camera_model")

            # 2. 尝试读取 ns-export cameras 导出的对齐相机数据
            cameras_json_path = self.cfg.project_dir / "cameras_export" / "transforms_train.json"
            
            if not cameras_json_path.exists():
                return None

            with open(cameras_json_path, 'r') as f:
                frames_list = json.load(f)
                
            webgl_poses = []
            for frame in frames_list:
                # ns-export cameras 输出为 3x4
                c2w_3x4 = np.array(frame['transform'])
                # 补成 4x4
                c2w = np.eye(4)
                c2w[:3, :4] = c2w_3x4
                
                # 注意：Three.js 的 Matrix4.fromArray 默认接受列优先 (Column-major) 数组
                # 所以这里必须用 .T 转置后再 flatten！
                c2w_threejs = c2w.T.flatten().tolist()
                
                # resolving image path
                file_path = frame.get('file_path')
                img_name = Path(file_path).name
                
                webgl_poses.append({
                    "id": img_name,
                    "fl_y": fl_y,
                    "h": h,
                    "matrix": c2w_threejs,
                    "image_url": f"/models/images/{img_name}"
                })
                
            # 对 webgl_poses 根据 id 自然排序
            import re
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', s['id'])]
            webgl_poses.sort(key=natural_sort_key)
            
            return webgl_poses
        except Exception as e:
            print(f"Error calculating WebGL poses: {e}")
            return None

    def _get_model_id(self, scene_id: str) -> Optional[str]:
        """根据 scene_id 获取 model_assets 表中的 id"""
        try:
            response = self.supabase.table("model_assets").select("id").eq("scene_id", scene_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]['id']
            return None
        except Exception as e:
            print(f"Error getting model_id: {e}")
            return None

    def _get_image_tag(self, image_path: Path) -> str:
        """调用 Qwen-VL 获取图片标签"""
        if not self.scene_analyzer.api_key:
            return ""
            
        prompt = "请用简短的中文描述这张图的拍摄视角和主要画面内容（例如：汽车正面特写、房间全景、从上方俯视），不要超过15个字。"
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.scene_analyzer._encode_image(str(image_path))}"}}
            ]}
        ]
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.scene_analyzer.api_key, base_url=self.scene_analyzer.base_url)
            completion = client.chat.completions.create(
                model=self.scene_analyzer.model, messages=messages, temperature=0.1
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting image tag: {e}")
            return ""
