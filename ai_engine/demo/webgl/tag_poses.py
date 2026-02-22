import os
import json
import base64
import random
from openai import OpenAI

# 使用百炼 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY", "your_api_key_here")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_tag(image_path):
    base64_image = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model="qwen-vl-max", # 使用多模态模型或者qwen-vl-plus
            messages=[
                {
                    "role": "user",
                    "content": [
                        # 提示词要求极简，方便用户搜索
                        {"type": "text", "text": "请用简短的中文描述这张图的拍摄视角和主要画面内容（例如：汽车正面特写、房间全景、从上方俯视），不要超过15个字。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"打标失败: {e}")
        return ""

def process_poses(poses_json_path, images_dir, output_path, sample_count=10):
    with open(poses_json_path, 'r', encoding='utf-8') as f:
        poses_data = json.load(f)
    
    # 支持两种格式：数组或者对象中带frames
    frames = poses_data.get('frames', poses_data) if isinstance(poses_data, dict) else poses_data
    
    sampled_frames = random.sample(frames, min(sample_count, len(frames)))
    
    for frame in frames:
        if frame in sampled_frames:
            # 解析图片路径
            image_url = frame.get('image_url')
            if not image_url:
                frame['tag'] = ""
                continue
                
            image_name = image_url.split('/')[-1]
            img_path = os.path.join(images_dir, image_name)
            
            if os.path.exists(img_path):
                print(f"正在分析视角 {image_name} ...")
                tag = get_image_tag(img_path)
                frame['tag'] = tag
                print(f"识别结果: {tag}")
            else:
                frame['tag'] = ""
        else:
            frame['tag'] = ""
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(poses_data, f, ensure_ascii=False, indent=2)
        
    print(f"打标完成！已保存至 {output_path}")

if __name__ == '__main__':
    images_dir = 'my-3dgs-viewer/public/models/images'
    poses_json = 'my-3dgs-viewer/public/models/webgl_poses.json'
    output_json = 'my-3dgs-viewer/public/models/webgl_poses_with_tags.json'
    
    if os.getenv("DASHSCOPE_API_KEY"):
        process_poses(poses_json, images_dir, output_json, sample_count=15)
    else:
        print("请在环境变量中设置 DASHSCOPE_API_KEY")
