# src/modules/scene_analyzer.py
# 功能：实现场景分析功能，使用Qwen-VL对图像进行质量评估和内容描述
# 实现：调用阿里云Qwen-VL大模型，分析图像质量并生成场景描述
# 逻辑：1. 随机抽选图像 2. 调用Qwen-VL进行分析 3. 生成质量评分和场景描述 4. 返回分析结果
# 包含：SceneAnalyzer类、图像编码方法、场景分析方法、质量评估算法
import base64
import json
import os
import random
import re
import ast
from pathlib import Path
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from src.config import PipelineConfig

class SceneAnalyzer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.api_key = self.cfg.dashscope_api_key
        self.base_url = self.cfg.dashscope_base_url
        # 优先使用视觉模型，避免误用文本模型导致“未提供任何图片”。
        self.model = self.cfg.dashscope_vl_model
        self.request_timeout = float(getattr(self.cfg, "dashscope_timeout_seconds", 45.0) or 45.0)
        self._client = None

    def _get_client(self):
        if OpenAI is None:
            raise RuntimeError("OpenAI client not available in environment")
        if self._client is None:
            # 显式设置超时与重试，避免网络抖动时长时间卡住。
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.request_timeout,
                max_retries=1,
            )
        return self._client

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def run(self, images_dir, log_callback=None):
        """
        返回: (passed: bool, score: int, reason: str, tags: list, description: str, objects: list)
        """
        if not self.api_key:
            return True, 60, "No API Key (Skipped)", [], "", []

        # 随机抽图逻辑：兼容更多后缀，并统一大小写处理
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = sorted(
            [p.name for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in image_exts]
        )
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
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(str(Path(images_dir) / f))}"}} for f in selected_files]
            ]}
        ]

        try:
            if log_callback: log_callback("🤖 [Qwen-VL] 正在进行场景评分与打标...")
            
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1
            )
            
            resp = str(completion.choices[0].message.content)
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            m = re.search(r"(\{.*\})", cleaned, flags=re.S)
            json_text = m.group(1) if m else cleaned

            # 兼容模型偶发的 JSON 格式偏差
            result = None
            parse_errors = []
            try:
                result = json.loads(json_text)
            except Exception as e1:
                parse_errors.append(str(e1))
                try:
                    result = json.loads(json_text.replace("'", '"'))
                except Exception as e2:
                    parse_errors.append(str(e2))
                    try:
                        result = ast.literal_eval(json_text)
                    except Exception as e3:
                        parse_errors.append(str(e3))

            if result is None:
                raise ValueError(f"Failed to parse model JSON output. Attempts: {parse_errors}. Raw: {resp}")
            
            # 解析结果
            score = int(result.get("score", 0))
            reason = result.get("reason", "Unknown")
            tags = result.get("tags", [])
            passed = score >= self.cfg.min_quality_score
            
            # 🟢 [新增] 提取描述信息
            description = result.get("description", "")
            objects = result.get("objects", [])

            # 模型未识别到图片时，避免错误阻断主流程
            no_image_markers = ("未提供任何图片", "没有图片", "no image", "no images")
            if isinstance(reason, str) and any(k in reason.lower() for k in [s.lower() for s in no_image_markers]):
                if log_callback:
                    log_callback("⚠️ [Qwen-VL] 未识别到图片输入，已自动跳过 AI 质检，继续后续流程")
                return True, 60, "模型未识别到图片，已跳过AI质检", [], description, objects
            
            return passed, score, reason, tags, description, objects

        except Exception as e:
            if log_callback: log_callback(f"⚠️ 分析出错: {e}")
            return True, 60, "Analysis Error (Default Pass)", [], "", []

    def analyze_single_image(self, image_path: str, log_callback=None) -> dict:
        """
        [新增] 针对单张图片的分析接口。

        返回值: dict 包含以下字段（尽量保持与多图 run 返回结构一致，方便上层使用）：
            {
                "score": int,
                "reason": str,
                "tags": list[str],
                "description": str,
                "objects": list[str]
            }

        如果 API Key 不存在或分析失败，返回空的默认结构（不会抛异常，保证流水线可继续）。
        """
        # 如果没有配置 API Key，直接返回空结构，避免抛错
        if not self.api_key:
            if log_callback:
                log_callback("⚠️ [SceneAnalyzer] 未配置 DASHSCOPE_API_KEY，跳过单图分析")
            return {"score": 0, "reason": "No API Key", "tags": [], "description": "", "objects": []}

        # 构造 Prompt（简洁，要求直接返回纯 JSON）
        prompt = """
        你是一个图像分析专家。请仔细分析这张图片并以纯 JSON 格式返回以下字段：
        1) score: 图片适合用于 3DGS 重建的质量分数（0-100）
        2) reason: 简短说明评分理由
        3) tags: 5-10 个关键词标签
        4) description: 对场景的详细自然语言描述
        5) objects: 图中主要物体列表

        请严格返回 JSON，形如：
        {"score": 85, "reason": "光照充足", "tags": ["室内","红色杯子"], "description": "...", "objects": ["红色杯子","桌子"]}
        """

        messages = [
            {"role": "system", "content": "You are a helpful image analysis assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"}}
            ]}
        ]

        try:
            if log_callback:
                log_callback(f"🤖 [Qwen-VL] 正在分析单张图片: {image_path}")

            client = self._get_client()
            completion = client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)

            resp = completion.choices[0].message.content
            # 清洗模型可能返回的代码块标记
            resp = str(resp).replace("```json", "").replace("```", "").strip()

            # 尝试提取第一个 JSON 对象（模型有时会返回多余文本）
            m = re.search(r"(\{.*\})", resp, flags=re.S)
            json_text = m.group(1) if m else resp

            # 多轮解析尝试：标准 json -> 替换单引号 -> ast.literal_eval
            result = None
            parse_errors = []
            try:
                result = json.loads(json_text)
            except Exception as e1:
                parse_errors.append(str(e1))
                try:
                    # 有些模型会用单引号或python dict格式返回
                    alt = json_text.replace("'", '"')
                    result = json.loads(alt)
                except Exception as e2:
                    parse_errors.append(str(e2))
                    try:
                        result = ast.literal_eval(json_text)
                    except Exception as e3:
                        parse_errors.append(str(e3))

            if result is None:
                raise ValueError(f"Failed to parse model JSON output. Attempts: {parse_errors}. Raw: {resp}")

            # 规范化输出字段
            return {
                "score": int(result.get("score", 0)),
                "reason": result.get("reason", ""),
                "tags": result.get("tags", []),
                "description": result.get("description", ""),
                "objects": result.get("objects", [])
            }

        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ [SceneAnalyzer] 单图分析出错: {e}")
            # 返回安全默认值，保证流水线继续
            return {"score": 0, "reason": "Analysis Error", "tags": [], "description": "", "objects": []}

    def select_best_preview(self, frames: list, images_dir: str, log_callback=None) -> tuple[int, str]:
        """
        [新增] 从一组带有位姿的帧中挑选出最适合作为封面（视角最全面）的图。
        返回: (最合适帧的原始索引, 选择理由)。
        """
        if not frames:
            return 0, "无可用帧，默认第 0 帧"
        if len(frames) == 1:
            return 0, "仅有 1 帧候选，默认选中"

        # 如果未配置 API Key
        if not self.api_key:
            if log_callback:
                log_callback("⚠️ [SceneAnalyzer] 未配置 API Key，封面图选择回退到第一帧")
            return 0, "未配置 API Key，回退第一帧"

        # 从帧列表中均匀抽取最多 8 张候选图
        sample_size = min(8, len(frames))
        indices = [int(i * (len(frames) - 1) / (sample_size - 1)) for i in range(sample_size)]
        candidate_frames = [frames[i] for i in indices]

        prompt = f"""
        你是一个 3D 场景分析助手。我提供了几张同一个场景的候选帧。
        请根据以下标准，选出最适合作为 3D 模型预览图（封面图）的一张：
        1. 视角最全面（不是看细节，而是看整体）。
        2. 如果是房间，选择能看到房间绝大部分内容的。
        3. 如果是单个物体，选择物体正面平视的。
        4. 如果是书桌等场景，选择能正着拍到整个主体的。

        请严格返回 JSON（不要 markdown 代码块）：
        {{
            "index": 0,
            "reason": "一句话说明选择理由"
        }}
        其中 index 必须是 0 到 {sample_size - 1} 的整数。
        """

        try:
            if log_callback:
                log_callback(f"🤖 [Qwen-VL] 正在从 {sample_size} 张候选图中挑选最佳封面...")

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]

            # 组装图片数据
            for frame in candidate_frames:
                # 兼容不同格式的图片路径记录
                # webgl_poses.json 使用了 'id' 或者 'image_url'，或者从 frames 取 'file_path'
                # 尽量兼容：
                img_name = frame.get('id') or frame.get('file_path') or frame.get('image_url')
                if not img_name:
                    continue
                # 处理可能包含 images/ 前缀的情况
                if img_name.startswith("images/"):
                    img_name = img_name[7:]
                elif img_name.startswith("images\\"):
                    img_name = img_name[7:]
                
                img_path = os.path.join(images_dir, img_name)
                # 简单容错：如果路径不存在，尝试去掉前缀或直接拼接
                if not os.path.exists(img_path):
                    if log_callback: log_callback(f"⚠️ 图片不存在: {img_path}")
                    pass # 但我们还是继续，可能会导致调用失败
                
                if os.path.exists(img_path):
                    messages[1]["content"].append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(img_path)}"}
                    })

            if len(messages[1]["content"]) == 1:
                return 0, "没有有效候选图片，回退第一帧"

            client = self._get_client()
            completion = client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)

            resp = str(completion.choices[0].message.content).strip()

            cleaned = resp.replace("```json", "").replace("```", "").strip()
            m_json = re.search(r"(\{.*\})", cleaned, flags=re.S)
            json_text = m_json.group(1) if m_json else cleaned

            selected_idx = None
            selected_reason = ""

            # 优先按 JSON 解析
            try:
                result = json.loads(json_text)
                selected_idx = int(result.get("index"))
                selected_reason = str(result.get("reason", "")).strip()
            except Exception:
                # 兼容旧格式：只返回数字
                m_idx = re.search(r"\d+", cleaned)
                if m_idx:
                    selected_idx = int(m_idx.group(0))

                # 容错提取理由（例如 "reason: xxx" 或 "理由：xxx"）
                m_reason = re.search(r"(?:reason|理由)\s*[:：]\s*(.+)", cleaned, flags=re.I)
                if m_reason:
                    selected_reason = m_reason.group(1).strip()

            if selected_idx is not None and 0 <= selected_idx < sample_size:
                best_orig_idx = indices[selected_idx]
                if not selected_reason:
                    selected_reason = "模型未提供明确理由"
                if log_callback:
                    log_callback(
                        f"✅ AI 成功选出封面帧: index {best_orig_idx} (候选图中的第 {selected_idx} 张)，理由: {selected_reason}"
                    )
                return best_orig_idx, selected_reason

            if log_callback:
                log_callback(f"⚠️ 无法解析 AI 的选择结果: '{resp}'，回退到第一帧")
            return 0, "无法解析模型输出，回退第一帧"

        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ 挑选封面失败: {e}，回退到第一帧")
            return 0, f"挑选失败: {e}"

    def select_best_image(self, image_paths: list[str], log_callback=None) -> tuple[int, str]:
        """
        从一组图片路径里选出“最适合快速重建”的单张图。
        返回: (index, reason)。
        """
        if not image_paths:
            return 0, "无候选图片，默认第 0 张"
        if len(image_paths) == 1:
            return 0, "仅 1 张候选图，默认选中"
        if not self.api_key:
            if log_callback:
                log_callback("⚠️ [SceneAnalyzer] 未配置 API Key，最佳帧选择回退到第一张")
            return 0, "未配置 API Key，回退第一张"

        prompt = f"""
        你是 3D 重建数据筛选助手。请从以下候选图中选择最适合做“快速重建输入”的一张。
        判断标准：
        1) 画面清晰、主体完整、无遮挡、无明显运动模糊；
        2) 如果是单物体，优先“正面、摆正、主体占比适中”的图；
        3) 如果是场景，优先“构图完整、能看到整体空间结构”的图。

        仅输出 JSON：
        {{
          "index": 0,
          "reason": "一句话理由"
        }}
        index 必须是 0 到 {len(image_paths) - 1} 的整数。
        """

        try:
            if OpenAI is None:
                raise RuntimeError("OpenAI client not available in environment")

            content = [{"type": "text", "text": prompt}]
            for path in image_paths:
                if os.path.exists(path):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(path)}"}
                    })

            if len(content) == 1:
                return 0, "无有效候选图，回退第一张"

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
            )

            resp = str(completion.choices[0].message.content).strip()
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            m = re.search(r"(\{.*\})", cleaned, flags=re.S)
            json_text = m.group(1) if m else cleaned

            idx = None
            reason = ""
            try:
                result = json.loads(json_text)
                idx = int(result.get("index"))
                reason = str(result.get("reason", "")).strip()
            except Exception:
                m_idx = re.search(r"\d+", cleaned)
                if m_idx:
                    idx = int(m_idx.group(0))
                m_reason = re.search(r"(?:reason|理由)\s*[:：]\s*(.+)", cleaned, flags=re.I)
                if m_reason:
                    reason = m_reason.group(1).strip()

            if idx is not None and 0 <= idx < len(image_paths):
                return idx, (reason or "模型未提供明确理由")

            return 0, "无法解析模型输出，回退第一张"
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ 最佳帧选择失败: {e}，回退第一张")
            return 0, f"选择失败: {e}"

    def classify_scene_or_object(self, image_path: str, log_callback=None) -> tuple[str, str]:
        """
        判定输入图是“场景(scene)”还是“单物体(object)”。
        返回: (label, reason)，label ∈ {'scene', 'object'}。
        """
        if not image_path or not os.path.exists(image_path):
            return "scene", "输入图不存在，默认按场景处理"
        if not self.api_key:
            return "scene", "未配置 API Key，默认按场景处理"

        prompt = """
        你是视觉分类助手。请判断图片更适合以下哪一类 3D 重建目标：
        - scene: 空间/房间/场景整体（可看到较大环境范围）
        - object: 单个主体物体（主体集中且应优先做单物体重建）

        仅输出 JSON：
        {
          "label": "scene",
          "reason": "一句话理由"
        }
        label 只能是 scene 或 object。
        """

        try:
            if OpenAI is None:
                raise RuntimeError("OpenAI client not available in environment")

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"}},
                    ]},
                ],
                temperature=0.1,
            )

            resp = str(completion.choices[0].message.content).strip()
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            m = re.search(r"(\{.*\})", cleaned, flags=re.S)
            json_text = m.group(1) if m else cleaned

            label = "scene"
            reason = ""
            try:
                result = json.loads(json_text)
                label = str(result.get("label", "scene")).strip().lower()
                reason = str(result.get("reason", "")).strip()
            except Exception:
                low = cleaned.lower()
                if "object" in low:
                    label = "object"
                if "scene" in low:
                    label = "scene" if "object" not in low else label
                m_reason = re.search(r"(?:reason|理由)\s*[:：]\s*(.+)", cleaned, flags=re.I)
                if m_reason:
                    reason = m_reason.group(1).strip()

            if label not in ("scene", "object"):
                label = "scene"
            return label, (reason or "模型未提供明确理由")
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ 场景/物体判定失败: {e}，默认按场景处理")
            return "scene", f"判定失败: {e}"
