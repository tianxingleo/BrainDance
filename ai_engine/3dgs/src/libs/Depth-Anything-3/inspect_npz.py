import numpy as np
import sys

try:
    data = np.load("/home/ltx/projects/Depth-Anything-3/output/sugar_streaming/results_output/frame_0.npz")
    print("Keys in npz:", data.files)
    for k in data.files:
        print(f"Key: {k}, Shape: {data[k].shape}, Dtype: {data[k].dtype}")
except Exception as e:
    print(e)
