-- BrainDance Realtime 测试种子数据

insert into public.processing_tasks (
  id, user_id, scene_id, status, display_name, task_type, logs, task_params, description
) values
  (
    '10000000-0000-0000-0000-000000000011',
    'it_user_a',
    'it_realtime_pending_001',
    'pending',
    'IT Realtime Pending 001',
    'video_3dgs',
    '[]'::jsonb,
    '{}'::jsonb,
    'integration realtime pending'
  ),
  (
    '10000000-0000-0000-0000-000000000012',
    'it_user_a',
    'it_realtime_processing_001',
    'processing',
    'IT Realtime Processing 001',
    'video_3dgs',
    '[{"msg":"开始处理"}]'::jsonb,
    '{}'::jsonb,
    'integration realtime processing'
  ),
  (
    '10000000-0000-0000-0000-000000000013',
    'it_user_a',
    'it_realtime_completed_001',
    'completed',
    'IT Realtime Completed 001',
    'video_3dgs',
    '[{"msg":"处理完成"}]'::jsonb,
    '{}'::jsonb,
    'integration realtime completed'
  ),
  (
    '10000000-0000-0000-0000-000000000014',
    'it_user_a',
    'it_realtime_failed_001',
    'failed',
    'IT Realtime Failed 001',
    'video_3dgs',
    '[{"msg":"处理失败"}]'::jsonb,
    '{}'::jsonb,
    'integration realtime failed'
  )
on conflict (id) do update set
  status = excluded.status,
  logs = excluded.logs,
  updated_at = now();
