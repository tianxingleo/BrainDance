from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "run_real_chain_debug.py"


def load_module():
    spec = importlib.util.spec_from_file_location("part18_run_real_chain_debug", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_recent_answer_prefers_short_natural_recent_summary():
    module = load_module()

    evidence = [
        {
            "scene_id": "scene_new",
            "display_name": "桌面场景",
            "description": "桌面上有触控笔、机械键盘和鼠标。",
            "objects": ["触控笔", "机械键盘", "鼠标"],
            "tags": ["桌面"],
            "created_at": "2026-03-22T10:00:00Z",
        },
        {
            "scene_id": "scene_old",
            "display_name": "书架角落",
            "description": "书架旁边有地球仪和手办。",
            "objects": ["书架", "地球仪", "手办"],
            "tags": ["学习角"],
            "created_at": "2026-03-20T10:00:00Z",
        },
    ]

    answer = module.build_recent_answer("我最近拍了什么？", evidence, [])

    assert answer == "最近拍到的主要有触控笔和机械键盘，以及书架和地球仪。"


def test_build_must_answer_focus_answer_keeps_focus_in_first_clause():
    module = load_module()

    evidence = [
        {
            "scene_id": "scene_desk",
            "display_name": "办公桌场景",
            "description": "办公桌上有显示器、笔记本电脑和机械键盘。",
            "objects": ["办公桌", "显示器", "笔记本电脑", "机械键盘"],
            "tags": ["桌面"],
            "created_at": "2026-03-22T09:00:00Z",
        }
    ]

    answer = module.build_must_answer_focus_answer(
        "最近拍到过什么办公桌上的东西？",
        evidence,
        ["办公桌"],
    )

    assert answer == "最近拍到过办公桌相关内容，能看到办公桌、显示器和笔记本电脑。"


def test_build_model_inventory_answer_is_humanized():
    module = load_module()

    evidence = [
        {
            "scene_id": "scene_a",
            "display_name": "scene_a",
            "description": "这是一个洛天依主题展台模型，带有屏幕与展示台。",
            "objects": ["展台", "屏幕"],
            "tags": ["模型"],
            "created_at": "2026-03-22T09:00:00Z",
        },
        {
            "scene_id": "scene_b",
            "display_name": "scene_b",
            "description": "白色书架角落模型，包含地球仪和词典。",
            "objects": ["书架", "地球仪", "词典"],
            "tags": ["模型"],
            "created_at": "2026-03-21T09:00:00Z",
        },
    ]

    answer = module.build_model_inventory_answer(evidence)

    assert answer == "最近生成过2个模型，主要包括洛天依主题展台模型和白色书架角落模型。"


def test_build_semantic_lookup_answer_is_readable():
    module = load_module()

    evidence = [
        {
            "scene_id": "scene_cs",
            "display_name": "scene_cs",
            "description": "书桌上有算法导论、笔记本电脑和白板。",
            "objects": ["算法导论", "笔记本电脑", "白板"],
            "tags": ["学习相关"],
            "created_at": "2026-03-22T09:00:00Z",
        }
    ]

    answer = module.build_semantic_lookup_answer(
        "最近拍到过计算机科学相关内容吗？",
        evidence,
        ["计算机科学", "算法", "笔记本电脑", "白板"],
    )

    assert answer == "有，整体偏计算机科学方向，常见内容包括算法导论、笔记本电脑和白板。"


def test_build_semantic_lookup_answer_can_fallback_to_description_terms():
    module = load_module()

    evidence = [
        {
            "scene_id": "scene_study",
            "display_name": "scene_study",
            "description": "桌面上有教材、词典和白板，整体像学习区。",
            "objects": ["桌面", "书堆"],
            "tags": ["学习角"],
            "created_at": "2026-03-22T09:00:00Z",
        }
    ]

    answer = module.build_semantic_lookup_answer(
        "有没有学习氛围比较强的内容？",
        evidence,
        ["学习氛围"],
    )

    assert answer == "有，整体偏理工学习氛围，常见内容包括白板、教材和词典。"


def test_is_model_inventory_query_skips_abstract_semantic_model_questions():
    module = load_module()

    assert module.is_model_inventory_query(
        "有没有偏理工一点的模型？",
        "object_lookup",
        [],
    ) is False


def test_infer_semantic_terms_from_question_backfills_empty_search_text():
    module = load_module()

    assert module.infer_semantic_terms_from_question("有没有偏理工一点的模型？") == ["理工"]


def test_infer_answer_route_supports_new_formatter_routes():
    module = load_module()

    assert module.infer_answer_route(
        query_class="recent_capture",
        special_answer_route="recent_answer_formatter",
    ) == "recent_answer_formatter"
    assert module.infer_answer_route(
        query_class="object_lookup",
        special_answer_route="must_answer_focus_formatter",
    ) == "must_answer_focus_formatter"
