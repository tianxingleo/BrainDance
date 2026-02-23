import json
import numpy as np

with open('/home/ltx/projects/BrainDance/ai_engine/demo/webgl/my-3dgs-viewer/public/models/webgl_poses.json', 'r') as f:
    poses = json.load(f)

c2w = np.array(poses['frames'][0]['matrix']).reshape(4,4).T
print("Original c2w from Nerfstudio (Frame 0):")
print(c2w)

cvToGl = np.diag([1, -1, -1, 1])
c2w_gl = c2w @ cvToGl

print("Camera Matrix in GL space (c2w_gl):")
print(c2w_gl)
print("Position:", c2w_gl[:3, 3])

