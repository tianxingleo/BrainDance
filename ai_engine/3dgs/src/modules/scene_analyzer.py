# src/modules/scene_analyzer.py
# 功能：实现场景分析功能，使用Qwen-VL对图像进行质量评估和内容描述
# 实现：调用阿里云Qwen-VL大模型，分析图像质量并生成场景描述
# 逻辑：1. 随机抽选图像 2. 调用Qwen-VL进行分析 3. 生成质量评分和场景描述 4. 返回分析结果
# 包含：SceneAnalyzer类、图像编码方法、场景分析方法、质量评估算法
import os
import base64
import json
import random
from openai import OpenAI
from src.config import PipelineConfig

class SceneAnalyzer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.api_key = self.cfg.dashscope_api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "qwen-vl-max"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def run(self, images_dir, log_callback=None):
        """
        返回: (passed: bool, score: int, reason: str, tags: list, description: str, objects: list)
        """
        if not self.api_key:
            return True, 60, "No API Key (Skipped)", [], "", []

        # 随机抽图逻辑 (保持不变)
        all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        if len(all_images) < 5: return False, 0, "图片过少", [], "", []
        selected_files = random.sample(all_images, min(6, len(all_images)))
        
        # 🟢 [修改 Prompt] 让 AI 不仅打分，还要做详细描述
        prompt = f"""
        你是一个 3D 建模专家。请分析这些图片，提取用于构建 RAG 知识库的元数据。

        请完成两个任务：
        1. **质量评估**：打分 (0-100) 并判断是否适合 3DGS 建图。
        2. **内容描述**：详细描述场景内容，包括主体物体、颜色、材质、环境背景。
        
        及格线：{self.cfg.min_quality_score} 分。
        
        请严格输出 JSON 格式：
        {{
            "score": 85,
            "reason": "光照充足，纹理清晰。",
            "tags": ["室内", "红色", "马克杯", "木桌"],
            
            // 👇 新增：RAG 专用字段
            "description": "一张深色的实木桌子上放着一个红色的陶瓷马克杯，杯子有反光，背景是模糊的办公室环境，光线来自左侧窗户。",
            "objects": ["红色马克杯", "实木桌子", "窗户"]
        }}
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(os.path.join(images_dir, f))}"}} for f in selected_files]
            ]}
        ]

        try:
            if log_callback: log_callback("🤖 [Qwen-VL] 正在进行场景评分与打标...")
            
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1
            )
            
            resp = completion.choices[0].message.content.replace("```json", "").replace("```", "")
            result = json.loads(resp)
            
            # 解析结果
            score = result.get("score", 0)
            reason = result.get("reason", "Unknown")
            tags = result.get("tags", [])
            passed = score >= self.cfg.min_quality_score
            
            # 🟢 [新增] 提取描述信息
            description = result.get("description", "")
            objects = result.get("objects", [])
            
            return passed, score, reason, tags, description, objects

        except Exception as e:
            if log_callback: log_callback(f"⚠️ 分析出错: {e}")
            return True, 60, "Analysis Error (Default Pass)", [], "", []
