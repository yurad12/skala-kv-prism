import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from kvprism.tools.web_search import web_search

query = "TurboQuant LLM serving benchmark"

# 1회차 호출 (Cache Miss 또는 기존 캐시 로드)
start_1 = time.time()
res_1 = web_search(query=query, max_results=2)
dur_1 = time.time() - start_1

# 2회차 호출 (반드시 Cache Hit 발생)
start_2 = time.time()
res_2 = web_search(query=query, max_results=2)
dur_2 = time.time() - start_2

cache_files = list(Path("outputs/cache").glob("*.json"))

print("=" * 50)
print(f"1회차 실행 시간: {dur_1:.4f}초 (결과: {len(res_1)}건)")
print(f"2회차 실행 시간: {dur_2:.4f}초 (결과: {len(res_2)}건)")
print(f"outputs/cache 생성된 JSON 파일 수: {len(cache_files)}개")
if cache_files:
    print(f"최근 생성된 캐시 파일: {cache_files[-1].name}")
print("=" * 50)