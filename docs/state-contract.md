# Graph State 계약

`src/kvprism/graph/state.py`는 모든 에이전트가 공유하는 인터페이스다. 노드마다
서로 다른 딕셔너리 필드를 만들지 말고 이 파일의 모델을 가져와 사용한다.

## 11개 상태 키의 담당 노드

| 상태 키 | 작성 노드 | 사용 노드 |
| --- | --- | --- |
| `request` | `app.py` | 모든 노드 |
| `research` | `research` | 관점별 노드, `synthesize` |
| `market_eval` | `market` | `judge`, `synthesize` |
| `stakeholder_eval` | `stakeholder` | `judge`, `synthesize` |
| `domain_eval` | `domain` | `judge`, `synthesize` |
| `judge` | `judge` | 그래프 라우터, `synthesize` |
| `retry_count` | 그래프 라우터 | 그래프 라우터 |
| `retry_targets` | `judge` 또는 라우터 | 관점별 노드 라우터 |
| `synthesis` | `synthesize` | `report`, `app.py` |
| `report` | `report` | `app.py` |
| `sources` | 기술 조사·관점별 노드 | `judge`, `synthesize`, `report` |

## 병렬 실행 규칙

세 관점 노드가 병렬로 실행되므로 `sources`에만 리듀서를 적용한다. 각 노드는
자신이 새로 발견한 출처만 반환해야 한다.

```python
return {
    "market_eval": result,
    "sources": newly_discovered_sources,
}
```

노드가 전체 출처 목록을 반환하면 팬인 과정에서 기존 인용이 중복된다. 나머지
상태 키는 담당 노드가 새 값으로 덮어쓴다.

## 인용 규칙

모든 `EvidenceClaim`은 `source_id` 한 개를 가진다. 해당 ID는 공유
`sources` 목록에 반드시 존재해야 한다. `Source.excerpt`에는 논문이나 웹
페이지에서 가져온 원문을 그대로 저장한다. 생성 모델의 요약문은 평가 결과
필드에 기록하며 `excerpt`에 넣지 않는다.

## 재실행 규칙

검증 노드는 항상 `market`, `stakeholder`, `domain` 세 관점을 모두 검사한다.
라우터는 `judge.failed_perspectives`를 `retry_targets`에 기록하고
`retry_count`를 증가시킨 뒤, 미달한 관점 노드만 다시 실행한다.
`retry_count == 1`이면 미달 항목이 남아 있어도 종합 단계로 진행하고, 남은
문제는 보고서의 한계로 기록한다.

## 최소 노드 반환 형태

```python
def market_node(state: GraphState) -> dict:
    ...
    return {"market_eval": result, "sources": new_sources}


def judge_node(state: GraphState) -> dict:
    result = JudgeResult(...)
    return {"judge": result, "retry_targets": result.failed_perspectives}
```

그래프 시작 시 `initial_state(PipelineInput(...))`를 사용하고, `app.py`가
최종 결과를 반환하기 전에 `validate_pipeline_output(final_state)`를 호출한다.
