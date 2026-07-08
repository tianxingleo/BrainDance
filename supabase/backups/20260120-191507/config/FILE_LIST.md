# 配置文件清单

## 包含的文件
- config.toml - Supabase CLI 配置文件
- seed.sql - 存储桶初始化脚本
- migrations/* - 所有数据库迁移脚本

## 需要手动复制的文件（包含敏感信息）
- .env - Python worker 环境变量（需脱敏后复制或手动处理）
- supabase/.env.1 - Supabase 启动输出（包含连接信息）

## 注意事项
1. .env 文件包含 API 密钥和密码，备份后请妥善保管
2. 恢复时需要重新配置 .env 文件
