from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig
from src.modules.scene_analyzer import SceneAnalyzer


def _build_completion(raw_text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw_text))]
    )


def test_analyze_single_image_normalizes_string_lists(monkeypatch):
    analyzer = SceneAnalyzer(PipelineConfig())
    analyzer.api_key = "test-key"
    monkeypatch.setattr(analyzer, "_build_image_data_url", lambda path, log_callback=None: "data:image/jpeg;base64,ZmFrZQ==")

    monkeypatch.setattr(
        analyzer,
        "_get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: _build_completion(
                        '{"score":"88","reason":"清晰","tags":"杯子, 桌子","description":"","objects":"杯子，木桌"}'
                    )
                )
            )
        ),
    )

    result = analyzer.analyze_single_image("/tmp/fake.png")

    assert result["ok"] is True
    assert result["score"] == 88
    assert result["tags"] == ["杯子", "桌子"]
    assert result["objects"] == ["杯子", "木桌"]
    assert result["description"] == "图中主要物体包括：杯子、木桌。"


def test_analyze_single_image_reports_failure_without_fake_zero_score(monkeypatch):
    analyzer = SceneAnalyzer(PipelineConfig())
    analyzer.api_key = "test-key"
    monkeypatch.setattr(analyzer, "_build_image_data_url", lambda path, log_callback=None: "data:image/jpeg;base64,ZmFrZQ==")

    monkeypatch.setattr(
        analyzer,
        "_get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: _build_completion("not-json")
                )
            )
        ),
    )

    result = analyzer.analyze_single_image("/tmp/fake.png")

    assert result["ok"] is False
    assert result["score"] is None
    assert result["tags"] == []
    assert "Analysis Error" in result["reason"]


def test_build_image_data_url_keeps_small_png_without_wrong_mime(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsmall")

    analyzer = SceneAnalyzer(PipelineConfig())
    url = analyzer._build_image_data_url(str(image_path))

    assert url.startswith("data:image/png;base64,")


def test_classify_scene_or_object_uses_built_data_url(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsmall")

    analyzer = SceneAnalyzer(PipelineConfig())
    analyzer.api_key = "test-key"

    expected_url = "data:image/png;base64,ZmFrZQ=="
    monkeypatch.setattr(analyzer, "_build_image_data_url", lambda path, log_callback=None: expected_url)

    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _build_completion('{"label":"object","reason":"主体集中"}')

    monkeypatch.setattr(
        "src.modules.scene_analyzer.OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        ),
    )

    label, reason = analyzer.classify_scene_or_object(str(image_path))

    assert label == "object"
    assert reason == "主体集中"
    assert captured["messages"][1]["content"][1]["image_url"]["url"] == expected_url
