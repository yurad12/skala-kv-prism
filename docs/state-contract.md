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

세 관점 노드가 병렬로 실행되므로 `sources`에만 `merge_sources` 리듀서를
적용한다. 각 노드는 자신이 새로 발견한 출처만 반환해야 한다. 같은 출처가
재실행에서 다시 들어오면 `source_id`를 기준으로 한 건만 남긴다. 같은 ID에
서로 다른 내용이 들어오면 인용 충돌로 처리해 실행을 중단한다.

```python
return {
    "market_eval": result,
    "sources": newly_discovered_sources,
}
```

노드가 전체 출처 목록을 반환하면 팬인 과정에서 기존 인용이 중복된다. 나머지
상태 키는 담당 노드가 새 값으로 덮어쓴다.

## 인용 규칙

모든 `EvidenceClaim`은 `source_id`를 가진다. 해당 ID는 공유 `sources`
목록에 반드시 존재해야 한다. `Source.excerpt`에는 논문이나 웹 페이지에서
가져온 원문을 저장한다.

## 검증 항목

`JudgeChecks`는 관점마다 다음 항목을 각각 기록한다.

- 기술별 긍정·부정 근거 균형
- 모든 주장에 대한 인용 포함 여부
- TRL 판단 근거 포함 여부

`PerspectiveJudgment.passed`는 위 항목이 모두 통과한 경우에만 `True`가 될 수
있다. 실패 결과에는 `issues`와 `retry_instruction`이 반드시 포함된다.

## 재실행 규칙

검증 노드는 항상 `market`, `stakeholder`, `domain` 세 관점을 모두 검사한다.
라우터는 `judge.failed_perspectives`를 `retry_targets`에 기록하고
`retry_count`를 증가시킨 뒤, 미달한 관점 노드만 다시 실행한다.
`retry_count == 1`이면 추가 재실행을 허용하지 않는다.

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
이 함수는 11개 상태 키가 모두 생성됐는지, 기술 식별자가 입력과 일치하는지,
인용이 등록된 출처를 가리키는지, 종합 매트릭스와 REFERENCE가 등록된 출처만
사용하는지 검사한다.
