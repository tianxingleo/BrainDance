# BrainDance 集成测试脚本骨架

本目录用于承载 Flutter-Supabase 集成测试的环境引导、种子数据、状态变更、函数冒烟与全量编排脚本。

当前已落地的是骨架入口，主要目标是：

1. 固定脚本命名与参数风格
2. 明确后续实现的责任边界
3. 让 `app/integration_test/` 可以直接按约定接入

建议按以下顺序继续补实现：

1. `bootstrap_supabase_test_env.sh`
2. `seed_supabase_test_data.sh`
3. `cleanup_supabase_test_data.sh`
4. `mutate_processing_task_status.sh`
5. `run_flutter_integration_tests.sh`
6. `run_edge_function_smoke_tests.sh`
7. `run_full_integration_suite.sh`
