"""Supervisor 실행·SQLite 복구·최종 산출물 검증 진입점."""

from __future__ import annotations

import argparse
import json
import sqlite3
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from kvprism.graph import state as payload_models
from kvprism.graph import supervisor_state as control_models
from kvprism.graph.state import PipelineInput
from kvprism.graph.supervisor_state import (
    RunLimits, SupervisorOutput, initial_supervisor_state, validate_supervisor_output,
)
from kvprism.graph.supervisor_build import build_supervisor_graph
from kvprism.rag.config import ROOT

DEFAULT_INPUT = ROOT / "configs/pipeline.yaml"


def load_pipeline_input(path: str | Path, *, replay: bool | None = None) -> PipelineInput:
    """YAML 입력을 기존 SW/HW 공용 모델로 검증한다."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("파이프라인 입력 설정은 YAML 객체여야 합니다")
    if replay is not None:
        data["replay"] = replay
    return PipelineInput.model_validate(data)


def checkpoint_serializer() -> JsonPlusSerializer:
    """우리 공용 Pydantic 모델만 명시적으로 체크포인트에서 복원한다."""
    classes = {
        (value.__module__, value.__name__)
        for module in (payload_models, control_models)
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, BaseModel)
    }
    return JsonPlusSerializer(allowed_msgpack_modules=classes)


def run_pipeline(request: PipelineInput | None = None, *, graph: Any | None = None,
                 trace_id: str | None = None, resume: bool = False,
                 limits: RunLimits | None = None, output_root: str | Path | None = None) -> SupervisorOutput:
    """새 실행과 미완료 체크포인트 재개를 구분하며 실패 초안을 성공으로 반환하지 않는다."""
    if resume and (not trace_id or request is not None or limits is not None):
        raise ValueError("재개에는 기존 trace_id만 지정하며 입력과 실행 상한을 변경하지 않습니다")
    if not resume and request is None:
        raise ValueError("새 실행에는 request가 필요합니다")
    trace_id = str(UUID(trace_id)) if trace_id else str(uuid4())
    publish_final = output_root is None
    output_root = Path(output_root) if output_root else ROOT / "outputs/runs"
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / trace_id
    directory.mkdir(parents=True, exist_ok=True)

    def execute(compiled):
        config = {
            "configurable": {"thread_id": trace_id},
            "metadata": {"trace_id": trace_id}, "run_id": uuid4(),
            "run_name": "kvprism-supervisor",
        }
        if resume:
            snapshot = compiled.get_state(config)
            if not snapshot.values or not snapshot.next or snapshot.values.get("status") != "running":
                raise ValueError("재개할 미완료 체크포인트가 없습니다")
            effective_limits = snapshot.values["limits"]
            initial = None
        else:
            if getattr(compiled, "checkpointer", None) is not None and compiled.get_state(config).values:
                raise ValueError("기존 trace_id입니다. 미완료 실행은 --resume으로 재개하세요")
            effective_limits = limits or RunLimits()
            initial = initial_supervisor_state(request, trace_id=trace_id, limits=effective_limits)
        config["recursion_limit"] = 2 * effective_limits.max_decisions + 5
        final = None
        try:
            final = compiled.invoke(initial, config)
            result = validate_supervisor_output(final)
            if publish_final and result.output is not None:
                # 검증을 통과한 산출물만 기존 공유 위치에 함께 게시한다.
                report = final["report"]
                paths = {
                    "source_markdown_path": ROOT / "outputs/report.source.md",
                    "markdown_path": ROOT / "outputs/report.md",
                    "pdf_path": ROOT / "outputs/RAG-OUTPUT.pdf",
                }
                for key, destination in paths.items():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = getattr(report, key).resolve()
                    if source != destination.resolve():
                        shutil.copyfile(source, destination)
                published = report.model_copy(update=paths)
                result = result.model_copy(update={
                    "output": result.output.model_copy(update={"report": published}),
                })
        except Exception as error:
            # 인증 값이 포함될 수 있는 외부 예외 문자열은 출력하지 않는다.
            reason = f"파이프라인 실행 오류: {type(error).__name__}"
            result = SupervisorOutput(status="failed", trace_id=trace_id, termination_reason=reason)
        summary = result.model_dump(mode="json")
        if final and final.get("report") is not None and result.status != "succeeded":
            summary["draft_pdf_path"] = str(final["report"].pdf_path)
        (directory / "run-result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    if graph is not None:
        return execute(graph)
    with sqlite3.connect(output_root / "checkpoints.sqlite", check_same_thread=False) as connection:
        saver = SqliteSaver(connection, serde=checkpoint_serializer())
        return execute(build_supervisor_graph(checkpointer=saver, output_root=output_root))


def main() -> int:
    """정상 종료 0, 미완료 2, 실행 실패 1로 보고한다."""
    parser = argparse.ArgumentParser(description="Supervisor 기반 KV cache 기술 평가")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--replay", action="store_true", default=None)
    parser.add_argument("--resume", metavar="TRACE_ID")
    args = parser.parse_args()
    if args.resume and (args.config is not None or args.replay is not None):
        parser.error("--resume은 --config/--replay와 함께 사용할 수 없습니다")
    # 실행 진입점에서만 사용자가 설정한 키를 우선하고, 도구 모듈은 덮어쓰지 않는다.
    load_dotenv(ROOT / ".env", override=True)
    try:
        result = run_pipeline(trace_id=args.resume, resume=True) if args.resume else run_pipeline(
            load_pipeline_input(args.config or DEFAULT_INPUT, replay=args.replay))
    except ValueError as error:
        print(str(error))
        return 1
    print(f"status={result.status} trace_id={result.trace_id}")
    if result.output is not None:
        print(result.output.report.pdf_path.resolve())
    else:
        print(result.termination_reason)
        print(f"실행 기록: {ROOT / 'outputs/runs' / result.trace_id}")
    return {"succeeded": 0, "incomplete": 2, "failed": 1}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
