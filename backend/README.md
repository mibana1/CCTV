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
