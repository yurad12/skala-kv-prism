"""테스트 공통 환경 설정."""

import os


# 단위 테스트가 로컬 .env를 읽더라도 LangSmith로 테스트 데이터를 전송하지 않는다.
os.environ["LANGSMITH_TRACING"] = "false"
