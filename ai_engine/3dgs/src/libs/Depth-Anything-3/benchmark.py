#!/usr/bin/env python3
"""
RTX 5070 性能测试
"""
import os
import torch
import time
from depth_anything_3.api import DepthAnything3

# 使用 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

device = torch.device('cuda')
test_image = 'assets/examples/SOH/000.png'

print('=' * 60)
print('RTX 5070 性能测试')
print('=' * 60)

# 测试不同模型
models = [
    ('depth-anything/DA3-SMALL', '80M'),
    ('depth-anything/DA3-BASE', '120M'),
    ('depth-anything/DA3-LARGE', '350M'),
]

for model_name, params in models:
    print(f'\n测试模型: {model_name.split("/")[-1]} ({params})')

    # 加载模型
    print('  加载模型...')
    start_time = time.time()
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    load_time = time.time() - start_time
    print(f'  加载完成: {load_time:.2f}s')

    # 预热
    print('  预热推理...')
    prediction = model.inference([test_image])
    torch.cuda.synchronize()
    print('  预热完成')

    # 多次推理
    print('  性能测试...')
    times = []
    for i in range(10):
        start = time.time()
        prediction = model.inference([test_image])
        torch.cuda.synchronize()
        times.append(time.time() - start)
        if i % 5 == 0:
            print(f'    第 {i+1} 次推理: {times[-1]*1000:.1f}ms')

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f'\n  推理统计:')
    print(f'    平均: {avg_time*1000:.1f}ms')
    print(f'    最快: {min_time*1000:.1f}ms')
    print(f'    最慢: {max_time*1000:.1f}ms')
    print(f'    FPS: {1/avg_time:.1f}')

    # 显存使用
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f'  显存使用: {allocated:.2f}GB / {reserved:.2f}GB')

    # 清理
    del model
    torch.cuda.empty_cache()

print('\n' + '=' * 60)
print('测试完成！')
print('=' * 60)
