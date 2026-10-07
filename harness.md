# KV Prism 팀 공통 작업 규칙

## 목적과 기준

현재 구현은 유정이 작성한 통합 초안이다. 팀원은 담당 영역을 실제로 검토·보완하고 자신이 수행한 변경을 커밋한다. 기존 코드의 작성자나 기여 내역을 다른 사람의 작업처럼 바꾸지 않는다.

- 전체 코드를 공통으로 공유하고, 아래 표는 수정 담당 범위를 뜻한다.
- 설계서의 과제 요구사항과 팀에서 확정한 공용 계약을 따른다. 문서와 코드가 다르면 차이를 설명하고 공유한다.
- Supervisor 구조를 유지한다. 기존 RAG 그래프로 되돌리거나 다른 패턴으로 임의 변경하지 않는다.
- README는 사용자의 요청 없이 수정하지 않는다.

## 브랜치와 통합

- 기존 RAG의 `main`과 Supervisor 통합 브랜치 `develop/agent-supervisor`를 구분한다.
- 현재 `feature/supervisor-implementation`의 미커밋 작업은 아직 공유된 기준 커밋이 아니다. 유정이 초안을 보존하고 통합 브랜치에 반영한 뒤 팀에 기준 커밋을 알린다.
- 모두 동일한 통합 초안 커밋에서 담당 브랜치를 만든다. 브랜치나 초안 반영 여부를 확인하지 않고 생성·병합하지 않는다.
- 팀원 PR의 대상은 `develop/agent-supervisor`이다. `main`으로 직접 합치지 않는다.
- 다른 팀원의 작업을 임의로 덮어쓰거나 브랜치·파일·커밋을 삭제하지 않는다.

## 담당 파일

경로는 저장소 루트 기준이다. 별도 표기가 없는 에이전트 파일은 `src/kvprism/agents/`, 프롬프트 파일은 `src/kvprism/prompts/`에 있다.

| 담당 | 브랜치 | 수정 담당 |
| --- | --- | --- |
| 유정 | `feat/supervisor-orchestration` | `app.py`, `agents/supervisor.py`, `prompts/supervisor.md`, `src/kvprism/graph/supervisor_build.py` |
| 진욱 | `feat/research-rag-review` | `agents/research.py`, `agents/judge.py`, `prompts/research.py`, `src/kvprism/rag/`, `src/kvprism/tools/rag_retrieve.py`, `configs/rag.yaml` |
| 민경 | `feat/perspective-agents-review` | `agents/common.py`, `agents/market.py`, `agents/stakeholder.py`, `agents/domain.py`, `src/kvprism/tools/web_search.py`, `src/kvprism/tools/fetch_and_summarize.py` |
| 승민 | `feat/report-quality-review` | `src/kvprism/graph/supervisor_state.py`, `agents/synthesize.py`, `agents/report.py`, `agents/report_rules.py`, `agents/report_quality.py`, `prompts/synthesize.md`, `prompts/report.md`, `prompts/report_quality.md`, `src/kvprism/tools/render_pdf.py`, `src/kvprism/assets/fonts/` |

### 공용 파일과 충돌 방지

- `src/kvprism/graph/state.py`: 기존 필드명과 타입을 유지한다. 변경이 필요하면 먼저 영향 범위를 팀에 공유한다.
- `supervisor_state.py`: 승민이 수정하며 노드 이름·제어 필드·반환 계약 변경은 유정 및 해당 담당자와 맞춘다.
- `report_quality.py`: 승민이 파일을 관리한다. 진욱은 근거성·충실성 검토 결과를 전달하고 같은 파일을 동시에 수정하지 않는다.
- `.env.example`, `pyproject.toml`, `uv.lock`, `configs/pipeline.yaml`: 유정이 변경을 취합한다. 의존성을 임의로 추가하지 않는다.
- 담당 범위 밖의 문제가 발견되면 위치·원인·필요한 변경을 알려 담당자와 조율한다.
- `tests/`는 전체 공유한다. 기존 테스트를 활용하며 새 테스트는 사용자 요청 없이 추가하지 않는다.

## 구현과 설명

- 작업 전에 무엇을 왜 바꾸는지 짧게 설명하고, 완료 후 변경 내용과 검증 결과를 알린다.
- 코드 주석과 설명은 한국어로 쓴다. 필요하면 공통 타입·기술 조사 결과 등 역할별로 구분한다.
- 기존 함수·파일을 활용하고 불필요한 추상화나 문서를 만들지 않는다.
- 노드는 담당 결과 키만 반환한다. 출처 수집 노드는 새로 등록할 `sources`만 반환한다.
- 에이전트 간 읽기·쓰기 계약과 반환 타입을 임의 변경하지 않는다.
- 모든 하위 에이전트는 Supervisor로 돌아온다. 다음 담당과 보완 작업은 Supervisor LLM이 선택하고 코드는 허용 행동·실행 상한·최신성·완료 조건을 검증한다.
- 재실행 횟수를 예전 RAG 방식의 고정 1회로 설명하지 않는다. 현재 실행 설정과 기록을 기준으로 한다.
- 품질 평가를 통과한 최신 보고서만 성공 출력으로 게시한다. 초안 PDF 생성은 최종 성공과 다르다.

## 보고서와 출력

- 등록된 논문·참고자료만 사용하고 사실과 인용을 보존한다. 새로운 외부 사실을 임의 추가하지 않는다.
- TRL은 대상별 검증 근거로 판단한다. 근거가 부족하면 `판정 불가`로 표시한다. 스키마·내부 호환값 등 구현 설명은 독자에게 노출하지 않는다.
- 다른 실험 환경과 baseline의 수치를 직접 우열 비교하지 않는다. 확인된 실험 조건을 함께 쓴다.
- 시장은 채택·제품화·생태계, 이해관계자는 주체별 기대와 부담, 도메인은 목표 환경의 적용 조건을 다룬다.
- 시사점은 기술별 검토 조건과 적용 전 확인사항으로 작성한다. 기술 간 우열을 정하지 않는다.
- SUMMARY는 번호가 있는 세 항목으로, 매트릭스는 짧은 확인 신호와 제약으로 작성한다. 반복을 줄인다.
- 보고서 디자인·색·글꼴은 사용자의 승인 없이 변경하지 않는다.
- 최종 공유 PDF 경로는 `outputs/RAG-OUTPUT.pdf`이다. 임의로 `output/` 등 다른 공유 경로를 만들지 않는다.
- 실행 기록·체크포인트·평가용 초안은 `outputs/runs/`, 과거 산출물은 `outputs/archive/`에서 관리한다.
- 파일 정리는 참조 여부를 확인한 뒤 수행한다. 필요한 자료는 삭제하지 않고 보관하며 무관한 사용자 파일은 건드리지 않는다.

## 실행과 검증

- 기존 테스트 중 변경 범위에 해당하는 검사를 실행한다. 실제 API 호출과는 구분해 결과를 보고한다.
- 전체 파이프라인은 사용자가 직접 실행한다. AI는 별도 실행 요청 없이는 호출하지 않고 아래 명령을 안내한다.

```bash
uv run python app.py --replay
```

- `--replay`도 보고서 작성 과정에서 LLM API를 호출한다.
- 완료 보고에는 수정 파일, 검증 결과, 남은 문제를 짧게 적는다. 실행하지 않은 PDF나 평가를 완료했다고 말하지 않는다.
- `.env`, API 키, 캐시, 생성 PDF, 체크포인트는 팀 코드 전달·커밋에서 제외한다. `.env.example`에는 실제 비밀값을 넣지 않는다.


# Git 작업 규칙

## Commit
- 가능한 작은 단위로 커밋한다.
- 최소 파일 단위, 가능하면 함수/클래스 단위로 분리한다.
- 서로 다른 목적의 변경은 한 커밋에 섞지 않는다.


### Commit Message
형식:
`Type: 한국어 설명`

예시:
- `Feat: GraphState 스키마 추가`
- `Feat: Source 병합 리듀서 구현`
- `Refactor: 관점별 평가 모델 구조 정리`
- `Fix: 재실행 횟수 검증 오류 수정`
- `Test: State 검증 테스트 추가`
- `Docs: 개발 규칙 문서 추가`

사용 Type:
- `Feat`
- `Fix`
- `Refactor`
- `Test`
- `Docs`
- `Chore`

## Commit 실행 규칙
- 코드 작성·수정은 AI/Codex를 활용할 수 있다.
- `git add`, `git commit`, `git push`, PR 생성은 사용자가 직접 수행한다.
- AI/Codex는 커밋 메시지와 PR 문구만 제안한다.

## Pull Request
- 제목은 한국어로 작성한다.
- 본문도 한국어로 작성한다.
- 변경 이유와 주요 변경 사항만 간단히 작성한다.

예시 제목:
`오케스트레이션 State 스키마 추가`

예시 본문:
## 변경 배경
병렬 에이전트가 공통 데이터 구조를 사용하도록 State 계약을 정의했습니다.

## 주요 변경
- Pydantic 기반 State 모델 추가
- 출처 병합 리듀서 추가
- 재실행 및 TRL 결과 구조 정의

## 구현 원칙
- 필요한 기능만 구현한다.
- 코드 양이 많다고 좋은 것이 아니며, 불필요한 추상화·헬퍼·검증 로직을 추가하지 않는다.
- 현재 요구사항에서 사용되지 않는 기능은 미리 구현하지 않는다.
- 같은 목적을 더 단순하게 구현할 수 있으면 단순한 방식을 우선한다.
- 함수와 클래스는 책임이 명확할 때만 분리한다.
- 기존 구조를 크게 바꾸기보다 필요한 범위만 수정한다.
- "나중에 쓸 수도 있음"을 이유로 코드를 추가하지 않는다.
- 설계서와 현재 작업 범위에 없는 기능은 임의로 확장하지 않는다.