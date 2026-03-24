import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from src.config import PipelineConfig
from src.core.factory import PipelineFactory
from src.modules.scene_analyzer import SceneAnalyzer

LOCAL_TASK_TYPE_CHOICES = ["video_dual_chain", "video_3dgs", "da3_feed_forward_3dgs"]
SLOW_PIPELINE_CHOICES = ["video_3dgs", "da3_feed_forward_3dgs"]


def _detect_total_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return float(total_mem) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _extract_candidate_frames(video_path: Path, out_dir: Path, sample_count: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = out_dir / "frame_%05d.jpg"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "fps=5,scale=1280:1280:force_original_aspect_ratio=decrease:flags=lanczos",
        "-q:v", "2",
        str(frame_pattern),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("未能从视频提取候选帧")

    if sample_count <= 1:
        return [frames[0]]
    if len(frames) <= sample_count:
        return frames

    step = (len(frames) - 1) / float(sample_count - 1)
    picked = [frames[int(round(i * step))] for i in range(sample_count)]
    return sorted(set(picked))


def _run_local_pipeline_once(
    task_type: str,
    input_path: Path,
    task_params: Dict[str, Any],
    work_dir: Path,
    scene_id: str,
) -> Tuple[Path, Dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "task_id": f"local_{int(time.time())}",
        "scene_id": scene_id,
        "user_id": "local_user",
        "work_root": work_dir,
        "log_callback": print,
    }
    pipeline = PipelineFactory.get_pipeline(task_type, context)
    final_model_path, metadata = pipeline.run(str(input_path), task_params)

    if not final_model_path:
        raise RuntimeError(f"Pipeline 未返回模型路径: {task_type}")

    model_path = Path(final_model_path)
    if not model_path.exists():
        raise RuntimeError(f"Pipeline 输出文件不存在: {model_path}")

    return model_path, (metadata or {})


def _run_local_video_dual_chain(
    video_file: Path,
    scene_id: str,
    task_params: Dict[str, Any],
    output_root: Path,
):
    sample_count = int(task_params.get("best_frame_sample_count", 8))
    slow_pipeline = str(task_params.get("slow_pipeline", "video_3dgs")).strip() or "video_3dgs"
    if slow_pipeline not in tuple(SLOW_PIPELINE_CHOICES):
        slow_pipeline = "video_3dgs"

    vram_threshold = float(task_params.get("sam3d_vram_threshold_gb", 25))
    frames_dir = output_root / "_dual_chain_frames"
    fast_work_dir = output_root / "fast_chain"
    slow_work_dir = output_root / "slow_chain"

    fast_ok = False
    slow_ok = False
    fast_error = None
    slow_error = None

    cfg = PipelineConfig()
    analyzer = SceneAnalyzer(cfg)

    print("🖼️ [DualChain] 提取候选帧...")
    candidate_frames = _extract_candidate_frames(video_file, frames_dir, sample_count)
    best_idx, best_reason = analyzer.select_best_image([str(p) for p in candidate_frames], log_callback=print)
    best_image = candidate_frames[max(0, min(best_idx, len(candidate_frames) - 1))]
    print(f"✅ [DualChain] 最佳帧: {best_image.name}（{best_reason}）")

    classify_label, classify_reason = analyzer.classify_scene_or_object(str(best_image), log_callback=print)
    print(f"🔍 [DualChain] 快链目标判定: {classify_label}（{classify_reason}）")

    fast_task_type = "single_image_sharp"
    if classify_label == "object":
        total_vram_gb = _detect_total_vram_gb()
        if total_vram_gb >= vram_threshold:
            fast_task_type = "single_image_sam3d"
            print(f"🧠 [DualChain] 显存 {total_vram_gb:.1f}GB >= {vram_threshold}GB，快链使用 SAM3D")
        else:
            print(f"⚠️ [DualChain] 显存 {total_vram_gb:.1f}GB < {vram_threshold}GB，快链降级为 SHARP")
    else:
        print("🏞️ [DualChain] 判定为场景，快链使用 SHARP")

    try:
        fast_params = dict(task_params)
        fast_params["scene_id"] = scene_id
        model_path, _ = _run_local_pipeline_once(
            task_type=fast_task_type,
            input_path=best_image,
            task_params=fast_params,
            work_dir=fast_work_dir,
            scene_id=scene_id,
        )
        fast_ok = True
        print(f"⚡ [DualChain] 快链完成: {model_path}")
    except Exception as e:
        fast_error = e
        print(f"⚠️ [DualChain] 快链失败，将继续执行慢链: {e}")

    try:
        slow_params = dict(task_params)
        slow_params["scene_id"] = scene_id
        model_path, _ = _run_local_pipeline_once(
            task_type=slow_pipeline,
            input_path=video_file,
            task_params=slow_params,
            work_dir=slow_work_dir,
            scene_id=scene_id,
        )
        slow_ok = True
        print(f"🐢 [DualChain] 慢链完成: {model_path}")
    except Exception as e:
        slow_error = e
        print(f"⚠️ [DualChain] 慢链失败: {e}")

    if not fast_ok and not slow_ok:
        raise RuntimeError(f"快慢双链均失败 | fast={fast_error} | slow={slow_error}")


def run_local_mode(
    video_file: Path,
    task_type: str,
    slow_pipeline: str,
    sample_count: int,
    sam3d_vram_threshold_gb: float,
    project_name: str,
):
    """本地单次运行模式"""
    if not video_file.exists():
        print(f"❌ 找不到本地文件: {video_file}")
        return

    print(f"💿 启动本地模式: {video_file.name} | task_type={task_type}")
    scene_id = project_name or f"local_{int(time.time())}"
    output_root = Path("temp_workspace") / scene_id / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    task_params: Dict[str, Any] = {
        "slow_pipeline": slow_pipeline,
        "best_frame_sample_count": sample_count,
        "sam3d_vram_threshold_gb": sam3d_vram_threshold_gb,
        "scene_id": scene_id,
    }

    if task_type == "video_dual_chain":
        _run_local_video_dual_chain(
            video_file=video_file,
            scene_id=scene_id,
            task_params=task_params,
            output_root=output_root,
        )
    else:
        model_path, _ = _run_local_pipeline_once(
            task_type=task_type,
            input_path=video_file,
            task_params=task_params,
            work_dir=output_root,
            scene_id=scene_id,
        )
        print(f"✅ 本地任务完成: {model_path}")
