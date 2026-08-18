# Backend

FastAPI API, CCTV media processing, AI inference, persistence, and external
adapters live here. Long-running frame processing must run outside the API
request path.

```bash
cp ../.env.example ../.env
uv sync
uv run pytest
uv run cctv
```

설정은 프로젝트 루트의 `.env`와 `CCTV_`/`HIPERWALL_` 환경변수에서
로드합니다. 시작 시 설정된 SQLite 경로를 생성하고 아직 적용되지 않은
마이그레이션을 순서대로 적용합니다.

애플리케이션 이벤트는 stdout과 `CCTV_LOG_PATH`에 한 줄당 하나의 JSON
객체로 기록합니다. 로그에는 `timestamp`, `level`, `service`, `logger`,
`message`, `event`와 작업별 컨텍스트가 포함되며 인증정보는 자동으로
마스킹됩니다. HTTP 요청 로그에는 method, path, status code, duration이
구조화 필드로 기록되며 query string과 header는 기록하지 않습니다. 실제
자격증명을 메시지 문자열에 직접 넣지 마십시오.

실행 모드는 기본 `dry_run`이며 외부 부작용을 허용하지 않습니다. Hiperwall
어댑터 등 외부 작업 경계에서는 `external_action_allowed()`를 호출해야 하며,
차단된 작업은 `external_action_skipped` 이벤트로 기록됩니다. CPU와 분석 FPS가
기본값이고, LIVE 모드는 Hiperwall 주소와 선택된 인증 방식의 필수값을 검증한
후에만 시작됩니다.

`GET /health`는 API 버전, 환경, 실행 모드와 SQLite 읽기 가능 여부, 스키마
버전, journal mode를 반환합니다. SQLite를 읽을 수 없으면 민감한 파일 경로나
오류 메시지를 응답에 포함하지 않고 HTTP 503과 `health_check_failed` 로그를
기록합니다. Hiperwall과 영상 중계 상태는 해당 어댑터가 구현될 때 점검 항목에
추가합니다.

로컬 영상은 `LocalVideoDecoder`로 API 요청 경로 밖에서 디코딩합니다. 디코더는
원본 프레임 인덱스와 타임스탬프를 유지하면서 `sample_fps`에 맞는 프레임만
전달하고, 원본 FPS가 더 낮을 때 프레임을 복제하지 않습니다. 프리페치 큐를
만들지 않아 한 번에 현재 프레임만 보유하며 컨텍스트 종료 시 OpenCV 자원을
반드시 해제합니다.

```python
from cctv.media import LocalVideoDecoder

with LocalVideoDecoder("sample.mp4", sample_fps=2) as decoder:
    print(decoder.metadata)
    for frame in decoder.frames():
        analyze(frame.image, frame.timestamp_seconds)
```
