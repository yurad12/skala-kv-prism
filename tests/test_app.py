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

    def invoke(self, state: dict, config=None) -> dict:
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


def test_run_pipeline_validates_graph_final_state(tmp_path) -> None:
    request = app.load_pipeline_input(app.DEFAULT_INPUT)
    graph = RecordingGraph({"request": request})

    result = app.run_pipeline(request, graph=graph, output_root=tmp_path)
    assert result.status == "failed"
    assert result.output is None
    assert graph.received["request"] == request
    assert graph.received["step_count"] == 0
    assert graph.received["status"] == "running"
    assert "retry_count" not in graph.received


def test_cli_prefers_project_env(tmp_path, monkeypatch):
    """외부 프로세스의 키가 프로젝트 설정을 가리지 않도록 한다."""
    import os
    import sys
    from kvprism.graph.supervisor_state import SupervisorOutput
    (tmp_path / '.env').write_text('OPENAI_API_KEY=project-test-value\n')
    monkeypatch.setenv('OPENAI_API_KEY', 'inherited-test-value')
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['app.py'])

    def run(request):
        assert os.environ['OPENAI_API_KEY'] == 'project-test-value'
        return SupervisorOutput(status='incomplete', trace_id='test', termination_reason='테스트 종료')

    monkeypatch.setattr(app, 'run_pipeline', run)
    assert app.main() == 2
