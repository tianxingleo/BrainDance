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
        返回: (passed: bool, score: int, reason: str, tags: list)
        """
        if not self.api_key:
            return True, 60, "No API Key (Skipped)", []

        # 随机抽图逻辑 (保持不变)
        all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        if len(all_images) < 5: return False, 0, "图片过少", []
        selected_files = random.sample(all_images, min(6, len(all_images)))
        
        # 🟢 [关键修改] 使用宽容的评分制 Prompt
        prompt = f"""
        你是一个 3D 建模专家。请评估这些视频截图是否适合进行 3D Gaussian Splatting 重建。
        
        请给出一个 0-100 的评分：
        - 80-100: 完美（光照充足，纹理丰富，清晰）
        - 60-79: 良好（有轻微瑕疵但不影响生成）
        - 40-59: 一般（环境较差/弱光/部分模糊，但勉强可用）
        - 0-39: 不可用（纯黑/纯白/全屏马赛克/完全无纹理）
        
        当前设定的及格线是 {self.cfg.min_quality_score} 分。
        只要不是完全无法使用的废片，请尽量给高分以通过检查。
        
        请返回 JSON 格式：
        {{
            "score": 45,                // 评分
            "reason": "光线较暗，且有轻微运动模糊，但物体轮廓可见，勉强通过。",
            "tags": ["室内", "弱光", "人像", "低纹理"] // 提取3-5个场景标签
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
            
            score = result.get("score", 0)
            reason = result.get("reason", "Unknown")
            tags = result.get("tags", [])
            
            # 🟢 [核心逻辑] 拿分数和配置里的阈值比
            passed = score >= self.cfg.min_quality_score
            
            return passed, score, reason, tags

        except Exception as e:
            if log_callback: log_callback(f"⚠️ 分析出错: {e}")
            return True, 60, "Analysis Error (Default Pass)", []
