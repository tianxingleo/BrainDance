import json

try:
    with open('/home/ltx/projects/BrainDance/ai_engine/demo/webgl/my-3dgs-viewer/public/models/webgl_poses.json', 'r') as f:
        data = json.load(f)

    for i, frame in enumerate(data.get('frames', [])):
        if i == 0:
            frame['tag'] = '正面特写'
        elif i == 10:
            frame['tag'] = '侧面'
        elif i == 20:
            frame['tag'] = '背面'

    with open('/home/ltx/projects/BrainDance/ai_engine/demo/webgl/my-3dgs-viewer/public/models/webgl_poses_with_tags.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    print("Tags successfully injected!")
except Exception as e:
    print(f"Error: {e}")
