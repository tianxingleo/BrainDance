-- BrainDance Agent / Search 测试种子数据

insert into public.model_assets (
  id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info
) values
  (
    '20000000-0000-0000-0000-000000000021',
    'it_agent_scene_001',
    'it_user_a',
    'integration agent asset',
    array['keyboard','monitor'],
    array['it','agent'],
    'it_user_a/it_agent_scene_001/output/point_cloud.ply',
    'it_user_a/it_agent_scene_001/output/preview.txt',
    '{"source":"integration_agent"}'::jsonb
  )
on conflict (id) do update set
  description = excluded.description,
  tags = excluded.tags;

insert into public.memory_poses (
  id, model_id, image_name, transform_matrix, tag, embedding
) values
  (
    '40000000-0000-0000-0000-000000000001',
    '20000000-0000-0000-0000-000000000021',
    'it_frame_001.png',
    '[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]'::jsonb,
    'desk',
    null
  )
on conflict (id) do update set
  tag = excluded.tag,
  transform_matrix = excluded.transform_matrix;
