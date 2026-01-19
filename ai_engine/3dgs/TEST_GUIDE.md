# 单图 SAM3D Pipeline 测试指南

## 架构说明

```
main.py → worker.py → factory.py → pipeline.run()
              ↓
        Supabase 任务队列
```

- **main.py**: 程序入口 (视频本地测试 / 启动 Worker)
- **worker.py**: 监听 Supabase，根据 task_type 下载资源并调用 Pipeline
- **factory.py**: 根据 task_type 返回对应 Pipeline

---

## 快速测试

### 方式一：本地视频测试
```bash
cd ai_engine/3dgs
conda activate gs_linux_backup
python main.py /path/to/video.mp4
```

### 方式二：启动 Worker 监听 Supabase 任务
```bash
cd ai_engine/3dgs
conda activate gs_linux_backup
python main.py
```

### 方式三：本地单图快速测试 (推荐)
```bash
cd ai_engine/3dgs
python tests/test_local_single_image.py
```

---

## 测试数据位置

```
ai_engine/3dgs/test_data/
└── images/
    └── test_image.png          # 测试图片
```

---

## Supabase 任务创建示例

在 Supabase SQL Editor 中执行：

```sql
-- 1. 上传测试图片到存储桶
-- 路径: {user_id}/{scene_id}/raw/image.png
-- 存储桶: braindance-assets

-- 2. 创建任务记录
INSERT INTO processing_tasks (
    id,
    scene_id,
    user_id,
    task_type,
    task_params,
    status,
    created_at
) VALUES (
    'sam3d_test_001',
    'sam3d_scene_001',
    'test_user',
    'single_image_sam3d',
    '{
        "quality": "high",
        "repo_path": "/home/ltx/projects/BrainDance/ai_engine/3dgs/src/libs/sam-3d-objects",
        "model_dir": "/home/ltx/projects/BrainDance/ai_engine/models/sam3d/checkpoints"
    }',
    'pending',
    NOW()
);
```

### 任务表字段说明

| 字段 | 说明 | 示例值 |
|------|------|--------|
| `id` | 任务唯一ID | `sam3d_test_001` |
| `scene_id` | 场景ID (用于文件命名) | `sam3d_scene_001` |
| `user_id` | 用户ID | `test_user` |
| `task_type` | 任务类型 | `single_image_sam3d` |
| `task_params` | JSON 参数 | 见上方 |
| `status` | 状态 | `pending` |

---

## Worker 支持的任务类型

| task_type | 下载资源 | 说明 |
|-----------|----------|------|
| `video_3dgs` | 视频 (video.mp4) | 传统视频转3DGS |
| `single_image_sam3d` | 图片 (image.png) | 单张图片生成3DGS |

---

## 完整测试流程

### 1. 本地快速测试 (单图)
```bash
cd ai_engine/3dgs
conda activate gs_linux_backup
python tests/test_local_single_image.py
```

### 2. Supabase 集成测试

**步骤 1:** 配置 `.env` 文件
```bash
cd ai_engine/3dgs
cat >> .env <<EOF
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-key
SUPABASE_BUCKET=braindance-assets
EOF
```

**步骤 2:** 在 Supabase 创建任务
```sql
INSERT INTO processing_tasks (id, scene_id, user_id, task_type, task_params, status)
VALUES ('test_001', 'test_scene', 'user1', 'single_image_sam3d', '{}', 'pending');
```

**步骤 3:** 上传测试图片到存储
```
存储路径: user1/test_scene/raw/image.png
存储桶: braindance-assets
```

**步骤 4:** 启动 Worker
```bash
python main.py
```

---

## 常见问题

### Q: 提示 "找不到 SAM3D 仓库"
A: 检查 `src/config.py` 中的 `sam3d_repo_path` 是否正确

### Q: 提示 "找不到模型文件"
A: 确保 `models/sam3d/checkpoints/` 目录下有 checkpoint 文件

### Q: Worker 收不到任务
A: 检查 Supabase 连接和任务状态是否为 `pending`

### Q: 如何查看日志
A:
- 本地模式：直接输出到控制台
- Worker 模式：日志同步到 `processing_tasks.logs` 字段

