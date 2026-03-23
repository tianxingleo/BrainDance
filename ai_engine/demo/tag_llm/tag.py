import os
import random
import base64
import json
import shutil
from openai import OpenAI

# ================= 配置区域 =================

# 【重要】请替换为你自己的阿里云百炼 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max" # 或 qwen-vl-plus

INPUT_IMAGE_FOLDER = "image"           # 输入图片文件夹
OUTPUT_DIR = "output_analyzed"         # 输出结果文件夹
SAMPLE_COUNT = 16                       # 初始抽取用于分析的图片数量（建议8-10张）

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# ==========================================

def encode_image_to_base64(image_path):
    """将本地图片文件转换为 Base64 编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def select_random_images(folder_path, count):
    """从文件夹中随机抽取图片，返回 [(filename, full_path), ...]"""
    if not os.path.exists(folder_path):
        return []
    
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    if len(all_files) == 0:
        return []

    # 如果图片够多，就随机抽；不够就全拿
    selected_files = random.sample(all_files, min(len(all_files), count))
    
    # 返回文件名和完整路径的元组列表
    return [(f, os.path.join(folder_path, f)) for f in selected_files]

def generate_smart_analysis_prompt(file_list):
        """
        [修改版] 生成 Prompt，增加了对拍摄手法、画质、动态物体的详细分析字段
        """
        filenames_str = ", ".join([f"'{f}'" for f in file_list])
    
        prompt = f"""
你是一个专业的 3D 摄影测量与计算机视觉专家。我将提供一组对同一个物理场景进行采样拍摄的图片。
目前的图片文件名为: [{filenames_str}]。

请对这组图片进行深度的视觉分析，目的是为了选择最合适的 3D Gaussian Splatting (3DGS) 重建算法。
你需要完成以下任务，并严格按 JSON 格式输出：

1. **去重与优选**:
     - 剔除模糊、重复或被严重遮挡（如手指出境）的废片。
     - 优选画质最好、视角最清晰的图片。

2. **整体拍摄策略分析 (Global Analysis)**:
     - **拍摄轨迹**: 判断是围绕一个中心物体拍摄 (Orbit)，还是站在原地向四周扫描 (Scan)？
     - **场景封闭性**: 如果是扫描，场景是封闭的房间还是开放区域？
     - **主体数量**: 是专注拍摄单个物体，还是多个分散物体？
     - **摄影技巧**: 是否包含倾斜摄影视角（Oblique，即从上向下约45度俯拍）？
     - **干扰物检测**: 是否有手指、拍摄者倒影、或移动的路人/车辆入镜？

3. **单图详细打标**: 对保留的图片进行详细描述。

**输出要求**:
- **必须使用中文**。
- **严禁**输出 markdown 格式，直接输出纯 JSON 字符串。
- JSON 结构必须严格遵守以下 schema：

{{
    "global_scene": {{
        "shooting_strategy": {{
            "mode": "枚举值: 'Orbiting' (环绕物体) 或 'Scanning' (向四周扫描)",
            "is_enclosed": "Boolean: true (封闭空间，如室内/死胡同) 或 false (开放空间)",
            "orbit_target": "枚举值: 'Single Object' (单一物体), 'Multiple Objects' (分散物体), 'Scene' (如果是扫描模式则填此项)",
            "is_oblique_photography": "Boolean: true (主要采用俯视倾斜拍摄) 或 false (主要是平视)",
            "estimated_focal_length": "枚举值: 'Wide' (广角/有畸变), 'Standard' (标准), 'Telephoto' (长焦/压缩感)"
        }},
        "quality_assessment": {{
            "noise_level": "枚举值: 'Low', 'Medium', 'High' (明显噪点)",
            "dynamic_obstacles": ["列出检测到的干扰物，如: '手指遮挡', '拍摄者影子', '移动行人'。如无则为空列表"],
            "lighting_condition": "描述光照，如：'漫反射均匀', '强直射光', '暗光'"
        }},
        "summary": "场景的整体中文描述，涵盖布局和拍摄难点。"
    }},
    "selected_images": [
        {{
            "filename": "原始文件名",
            "quality_score": "1-10的评分",
            "reason_for_selection": "保留理由",
            "has_artifacts": "Boolean: 是否有轻微干扰但仍可接受",
            "description": "详细中文描述，用于 RAG 检索。"
        }}
    ]
}}
"""
        return prompt

def analyze_images_smartly(image_data_list):
    """
    发送图片和 Prompt 给大模型
    image_data_list: [(filename, path), ...]
    """
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    # 1. 提取纯文件名列表用于 Prompt
    filenames = [item[0] for item in image_data_list]
    
    # 2. 构建 User Message
    user_content = [
        {"type": "text", "text": "请根据以下提供的图片进行去重、优选和中文分析。注意每张图片对应的顺序。"}
    ]

    # 3. 添加图片 Payload
    # 注意：多模态模型通常按传入顺序对应图片，我们需要在 Prompt 里明示或依靠顺序
    # 这里我们在 Prompt 里给出了文件名列表，模型通常能按顺序对号入座（即第1张图对应列表第1个文件名）
    print(f"正在上传 {len(image_data_list)} 张图片给大模型进行分析...")
    
    for filename, path in image_data_list:
        b64 = encode_image_to_base64(path)
        if b64:
            user_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            # 也可以选择在每张图前加一个 text 提示文件名，但这会消耗更多 token 且部分模型不支持图文混排太碎
            # Qwen-VL 目前对顺序理解较好

    messages = [
        {"role": "system", "content": generate_smart_analysis_prompt(filenames)},
        {"role": "user", "content": user_content}
    ]

    try:
        print("等待模型思考与筛选 (Thinking)...")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1, # 低温度保证格式稳定
        )
        content = completion.choices[0].message.content
        
        # 清理 markdown
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
        
    except Exception as e:
        print(f"LLM 调用或解析失败: {e}")
        # print("Raw content:", content) # 调试用
        return None

def main():
    # 1. 准备目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # 清空旧数据（可选）
    os.makedirs(OUTPUT_DIR)

    # 2. 读取图片
    input_images = select_random_images(INPUT_IMAGE_FOLDER, SAMPLE_COUNT)
    if not input_images:
        print("未找到图片。")
        return

    # 3. LLM 分析
    result_json = analyze_images_smartly(input_images)
    if not result_json:
        return

    # 4. 处理结果
    global_info = result_json.get("global_scene", {})
    selected_list = result_json.get("selected_images", [])

    print(f"\n✅ 分析完成！")
    print(f"原始图片数: {len(input_images)} -> 优选后图片数: {len(selected_list)}")
    
    # 保存全局场景 JSON
    global_json_path = os.path.join(OUTPUT_DIR, "scene_summary.json")
    with open(global_json_path, 'w', encoding='utf-8') as f:
        json.dump(global_info, f, indent=2, ensure_ascii=False)
    print(f"全局分析已保存: {global_json_path}")

    # 5. 遍历优选列表，保存图片和对应 JSON
    # 建立一个 filename -> full_path 的映射，方便查找
    file_map = {fname: fpath for fname, fpath in input_images}

    for item in selected_list:
        fname = item.get("filename")
        if fname not in file_map:
            print(f"⚠️ 警告: 模型返回的文件名 '{fname}' 不在原始列表中，跳过。")
            continue
            
        src_path = file_map[fname]
        dst_path = os.path.join(OUTPUT_DIR, fname)
        json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}.json")

        # 复制图片
        shutil.copy2(src_path, dst_path)

        # 保存单张图片的 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
            
        print(f"  [保留] {fname} -> 生成描述 JSON")

    print(f"\n🎉 所有数据已处理完毕，请查看 '{OUTPUT_DIR}' 文件夹。")

if __name__ == "__main__":
    if API_KEY == "这里填写你的apikey":
        print("❌ 请先设置 API_KEY")
    else:
        main()