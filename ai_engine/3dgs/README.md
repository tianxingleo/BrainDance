BrainDance/
├── main.py                    # [入口] 实例化主 Pipeline 并运行
├── config.yaml
├── src/
    ├── __init__.py
    ├── core/                  # [核心数据]
    │   ├── __init__.py
    │   └── context.py         # 定义 PipelineContext (包含 3D 和 RAG 的所有数据字段)
    │
    ├── modules/               # [原子工具] (只会干具体的活，不知道流程)
    │   ├── __init__.py
    │   ├── reconstruction/    # 3D 相关工具
    │   │   ├── colmap.py
    │   │   └── nerfstudio.py
    │   └── rag/               # RAG 相关工具
    │       ├── vector_db.py
    │       ├── llm_client.py
    │       └── text_splitter.py
    │
    └── pipelines/             # [流程编排] (负责组装工具)
        ├── __init__.py
        ├── base.py            # [基类] 定义所有 Pipeline 的标准行为
        ├── main_pipeline.py   # [总指挥] BrainDance 总流程
        │
        └── sub_pipelines/     # [子流程]
            ├── __init__.py
            ├── recon_pipe.py  # 3D 重建子流水线
            └── rag_pipe.py    # RAG 知识库子流水线


BrainDance/
├── main.py                    # [入口] 只留最后那十几行启动代码
├── src/
│   ├── __init__.py
│   ├── config.py              # [配置] 存放 PipelineConfig
│   ├── core/                  # [核心]
│   │   ├── __init__.py
│   │   └── pipeline.py        # [流程] 存放 run_pipeline 函数
│   ├── modules/               # [业务类] 存放那几个大 Class
│   │   ├── __init__.py
│   │   ├── ai_segmentor.py    # 存放 AISegmentor + get_central_object_prompt
│   │   ├── glomap_runner.py   # 存放 GlomapRunner
│   │   ├── image_proc.py      # 存放 ImageProcessor
│   │   └── nerf_engine.py     # 存放 NerfstudioEngine
│   └── utils/                 # [工具函数] 存放 def 开头的纯算法函数
│       ├── __init__.py
│       ├── common.py          # 存放 format_duration
│       ├── cv_algorithms.py   # 存放 clean_and_verify_mask, get_salient_box
│       ├── geometry.py        # 存放 analyze_and_calculate_adaptive_collider
│       └── ply_utils.py       # 存放 perform_percentile_culling