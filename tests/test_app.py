"""전체 파이프라인 실행 진입점 테스트."""

from pathlib import Path

import pytest

import app
from kvprism.graph.state import PipelineInput


class RecordingGraph:
    """app.py가 만든 초기 State를 기록하는 컴파일 그래프 대역."""

    def __init__(self, final_state: dict):
        self.final_state = final_state
        self.received = None

    def invoke(self, state: dict) -> dict:
        self.received = state
        return self.final_state


def test_default_pipeline_config_creates_valid_input() -> None:
    request = app.load_pipeline_input(app.DEFAULT_INPUT)

    assert isinstance(request, PipelineInput)
    assert request.domain == "데이터센터"
    assert {technology.kind for technology in request.technologies} == {
        "software",
        "hardware",
    }


def test_replay_argument_overrides_config_value(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text(Path(app.DEFAULT_INPUT).read_text(encoding="utf-8"), encoding="utf-8")

    request = app.load_pipeline_input(config, replay=True)

    assert request.replay is True


def test_run_pipeline_validates_graph_final_state() -> None:
    request = app.load_pipeline_input(app.DEFAULT_INPUT)
    graph = RecordingGraph({"request": request})

    with pytest.raises(ValueError, match="그래프 출력에 필수 필드가 없습니다"):
        app.run_pipeline(request, graph=graph)

    assert graph.received == {
        "request": request,
        "retry_count": 0,
        "retry_targets": [],
        "sources": [],
    }
