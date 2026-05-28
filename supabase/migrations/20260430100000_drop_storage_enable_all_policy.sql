-- P0 修复: 删除 Storage "Enable all storage access" 全开策略
-- 该策略允许 anon 对所有 bucket 执行 ALL 操作，覆盖了前面所有 user folder 保护策略
-- 删除后，用户仍然可以通过已有的 "Allow user view/upload/delete own folder" 策略访问自己的文件
-- Worker 使用 service_role 绕过 RLS，不受影响

DROP POLICY IF EXISTS "Enable all storage access" ON storage.objects;
