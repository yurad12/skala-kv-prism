"""KV Prism 전체 LangGraph 파이프라인 실행 진입점."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from kvprism.graph.build import build_graph
from kvprism.graph.state import (
    PipelineInput,
    PipelineOutput,
    initial_state,
    validate_pipeline_output,
)
from kvprism.rag.config import ROOT


DEFAULT_INPUT = ROOT / "configs" / "pipeline.yaml"


def load_pipeline_input(path: str | Path, *, replay: bool | None = None) -> PipelineInput:
    """YAML 실행 설정을 읽고 외부 입력 스키마로 검증한다."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("파이프라인 입력 설정은 YAML 객체여야 합니다")
    if replay is not None:
        data["replay"] = replay
    return PipelineInput.model_validate(data)


def run_pipeline(request: PipelineInput, *, graph: Any | None = None) -> PipelineOutput:
    """초기 State로 그래프를 실행하고 최종 출력을 검증한다."""

    compiled = graph or build_graph()
    final_state = compiled.invoke(initial_state(request))
    return validate_pipeline_output(final_state)


def main() -> int:
    """명령행 입력을 읽어 보고서를 생성하고 최종 PDF 경로를 출력한다."""

    parser = argparse.ArgumentParser(description="KV cache 기술 다관점 평가 보고서 생성")
    parser.add_argument("--config", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--replay",
        action="store_true",
        default=None,
        help="저장된 research 응답과 검색 캐시를 재사용합니다",
    )
    args = parser.parse_args()

    # LangChain과 LangGraph가 OPENAI 및 LANGSMITH 표준 환경변수를 읽기 전에 로드한다.
    load_dotenv(ROOT / ".env", override=True)
    request = load_pipeline_input(args.config, replay=args.replay)
    output = run_pipeline(request)
    print(output.report.pdf_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
