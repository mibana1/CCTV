# CCTV AI Analysis and Hiperwall Integration

CCTV 영상을 수신해 객체를 분석하고 이벤트 메타데이터를 저장하며,
HiperInterface를 통해 Hiperwall 표시를 제어하기 위한 프로젝트입니다.
현재 개발 PC에서는 CPU 기반 1~3채널 소규모 검증만 수행하고,
GPU 및 다채널 검증은 별도의 동작 테스트 PC에서 진행합니다.

## 저장소 구조

- `backend`: FastAPI, DB, 분석, 미디어 파이프라인, 외부 연동 코드
- `frontend`: 관리 화면 영역. 현재는 기술 선택 전 최소 실행 뼈대
- `infra`: MediaMTX와 개발·배포 보조 설정
- `artifacts/models`: 외부에서 공급하는 모델 가중치. Git에서 제외
- `samples`: Git에 추가 가능한 작은 테스트 자료
- `runtime`: DB, 로그, 스냅샷, 임시 영상. Git에서 제외
- `docs/OPEN_QUESTIONS.md`: 미정 사항의 단일 관리 문서
- `docs/decisions`: 확정된 아키텍처 결정 기록

## 핵심 경계

- Hiperwall은 화면 표시와 레이아웃 제어만 담당합니다.
- Backend가 영상 수신, 분석, 저장, 검색, 규칙 판단을 담당합니다.
- HiperInterface로 프레임마다 객체 박스를 그리지 않습니다.
- 객체 박스가 필요하면 Backend 미디어 파이프라인이 별도의 분석 RTSP를 발행합니다.
- API 요청 처리 중에는 디코딩이나 AI 추론을 직접 수행하지 않습니다.

## Backend 시작

Ubuntu/WSL에서 다음을 실행합니다.

```bash
cd "/mnt/c/Users/노주형/Desktop/CCTVTest/CCTV"
cp .env.example .env
cd "/mnt/c/Users/노주형/Desktop/CCTVTest/CCTV/backend"
uv sync
uv run pytest
uv run cctv
```

API 확인 주소는 `http://127.0.0.1:8000/health`입니다.
애플리케이션은 프로젝트 루트의 `.env`와 운영체제 환경변수를 읽으며,
시작할 때 SQLite 파일과 미적용 스키마 마이그레이션을 초기화합니다.
기본 SQLite 경로는 `runtime/cctv.db`입니다.

Backend 애플리케이션 로그는 stdout과 `runtime/logs/cctv.jsonl`에 JSONL로
기록됩니다. 기본 회전 기준은 10MiB와 백업 5개이며 `.env`에서 조정할 수
있습니다. RTSP 사용자정보, 비밀번호, Token, Authorization, Cookie와 인증
XML 필드는 로그 출력 전에 마스킹됩니다.

## 실행 모드

기본 실행 모드는 `CCTV_APP_MODE=dry_run`, `CCTV_AI_DEVICE=cpu`,
`CCTV_ANALYSIS_FPS=2`입니다. DRY RUN은 영상 처리와 DB·로그 기록은 허용하지만
Hiperwall 같은 외부 시스템의 실제 변경 작업은 차단합니다. 외부 어댑터는
반드시 `cctv.core.execution.external_action_allowed`를 통과해야 합니다.

`CCTV_APP_MODE=live`는 `HIPERWALL_BASE_URL`이 있어야 하며, 인증 모드가
`token`이면 비어 있지 않은 `HIPERWALL_TOKEN`도 필요합니다. 조건이 맞지
않으면 설정 검증 단계에서 애플리케이션 시작이 실패합니다. CUDA 장치의 실제
사용 가능 여부는 추론 Worker 구현 단계에서 별도로 검사합니다.

### 규칙 이벤트 → Hiperwall DRY RUN

`CCTV_APP_MODE=dry_run`과 `CCTV_HIPERWALL_DRY_RUN_ENABLED=true`이면 로컬·RTSP
Worker가 규칙 이벤트를 Hiperwall 작업으로 변환합니다. `started`·`occurred`
이벤트는 `open_source`, `ended` 이벤트는 `restore_layout`으로 계획되며 실제
HTTP 요청은 보내지 않습니다. 작업은 규칙 이벤트와 같은 SQLite 트랜잭션에서
`display_actions`에 `simulated` 상태로 저장됩니다.

```bash
curl "http://127.0.0.1:8000/hiperwall-actions?status=simulated"
curl "http://127.0.0.1:8000/hiperwall-actions?analysis_run_id=<RUN_ID>"
```

응답의 `result.external_request_sent`는 항상 `false`입니다. 현재 구현은 DRY RUN
전용이며 LIVE 모드에서는 이 계획기를 활성화하지 않습니다.

## Docker 시작

저장소 루트에서 다음을 실행합니다.

```bash
cp .env.example .env
docker compose up --build
```

## RTSP 소규모 실행

MediaMTX가 외부 RTSP를 `camera` 경로로 한 번만 수신하고, CPU 분석 워커와
Hiperwall이 이 중계 경로를 각각 읽습니다. 실제 카메라가 없어도 로컬 영상을
반복 발행하는 `rtsp-test` 프로파일로 전체 수신·분석·DB 저장 흐름을 확인할 수
있습니다.

```bash
CCTV_RTSP_INPUT_URL=publisher \
docker compose --profile rtsp-test up -d mediamtx rtsp-test-publisher

CCTV_DETECTOR_ENABLED=true \
docker compose --profile rtsp run --rm --build rtsp-worker \
  --max-samples 10 --save-snapshots
```

실제 카메라를 사용할 때만 Git에서 제외된 `.env`에
`CCTV_RTSP_INPUT_URL=rtsp://username:password@camera-host:554/stream`을 설정합니다.
Hiperwall에서 원본 영상을 읽을 주소는
`rtsp://<Docker 호스트 IP>:8554/camera`입니다.

### 로컬 RTSP 테스트 대시보드

터미널의 JSON 로그를 직접 읽지 않고 RTSP 테스트를 시작·중지하고 진행률,
CPU·메모리, 얼굴 판정, 실행 요약과 스냅샷을 확인할 수 있습니다. 대시보드는
`CCTV_TEST_DASHBOARD_ENABLED=true`, 개발 환경, `dry_run`, localhost 요청일 때만
접근할 수 있습니다.

```bash
docker compose up -d --build mediamtx backend
```

브라우저에서 `http://127.0.0.1:18000/test-dashboard`를 엽니다. 카메라 이름,
위치, RTSP 주소와 계정을 입력하면 고유 MediaMTX 경로가 자동 생성되고 같은
화면에서 실시간 영상을 확인할 수 있습니다. 플레이어는 로컬 전용 HLS 포트
`http://127.0.0.1:18888/<자동 경로>`를 사용합니다. API 응답에는 계정을 포함하지
않고, SQLite의 원본 RTSP 주소는 Fernet으로 암호화합니다. 암호화 키는 기본적으로
Git에서 제외된 `runtime/secrets/camera_credentials.key`에 생성되므로 DB와 함께
백업해야 합니다.
한 번에 하나의
테스트 세션만 실행되며, 시작 API는 Backend 컨테이너 안에 별도 RTSP 워커
프로세스를 생성합니다. Docker 소켓이나 카메라 인증정보를 브라우저에 노출하지
않습니다. 테스트 종료 시 얼굴 비교 결과는 `face_match_events`에 저장되고,
스냅샷은 `runtime/snapshots/test-session-<ID>`에 남습니다.

```text
POST /test-sessions
POST /test-sessions/{session_id}/stop
GET  /test-sessions/{session_id}
GET  /test-sessions/{session_id}/events
GET  /test-sessions/{session_id}/snapshots
GET  /face-match-events?analysis_run_id={analysis_run_id}
GET  /analysis-runs/{analysis_run_id}/face-match-summary
POST /test-cameras
GET  /test-cameras
PATCH /test-cameras/{camera_id}
DELETE /test-cameras/{camera_id}
POST /test-cameras/{camera_id}/sync
GET  /test-cameras/{camera_id}/status
```

객체 분석기는 `CCTV_DETECTOR_TYPE`으로 선택하며 기본값은 `yolo_onnx`입니다.
공통 `ObjectDetector` 인터페이스를 구현해 등록하면 워커·추적·규칙·DB 코드를
바꾸지 않고 다른 객체 검출 모델로 교체할 수 있습니다. 기존
`CCTV_YOLO_ENABLED`도 호환되지만 신규 설정은 `CCTV_DETECTOR_ENABLED=true`를
사용합니다.

객체 분석 시에는 기본으로 클래스별 IoU 객체 추적이 적용되어 각 검출 결과에
분석 실행 내에서 유효한 `track_id`가 저장됩니다. `/detections` 응답에서도 ID를
확인할 수 있고, 특정 객체의 이력은
`/detections?analysis_run_id=<run-id>&track_id=<track-id>`로 조회합니다. 추적을
끄려면 `CCTV_TRACKING_ENABLED=false`를 사용합니다. 기본값은 IoU 임계값 `0.3`,
누락 허용 `4`프레임, 유휴 만료 `3`초이며 각각 `.env`에서 조정할 수 있습니다.
과거 스키마에서 저장된 검출 결과의 `track_id`는 `null`로 유지됩니다.

스키마 v4부터 추적 결과는 `tracks`와 `track_observations`에도 정규화해
저장합니다. `tracks`에는 최초·최종 관측 시점, 관측 수, 최대 신뢰도가 누적되고,
`track_observations`는 각 관측을 원본 검출과 프레임에 연결합니다. 기존에
`detections.track_id`가 저장된 데이터는 마이그레이션 시 자동으로 백필됩니다.

- 트랙 목록: `GET /tracks?analysis_run_id=<run-id>&min_observations=2`
- 트랙 상세: `GET /analysis-runs/<run-id>/tracks/<track-id>`
- 관측 이력: `GET /analysis-runs/<run-id>/tracks/<track-id>/observations`

트랙 목록은 클래스, 영상 소스 기준 관측 시간, 현재 활성 상태를 함께 필터링할 수
있습니다. 시간 범위는 트랙의 최초·최종 관측 구간과 요청 구간이 겹치는 항목을
반환합니다. `active=true`는 추적기가 누락 허용 범위 안에서 아직 유지 중인 ID를
의미하며, 분석 실행이 끝나면 해당 실행의 모든 트랙은 비활성화됩니다.

```text
GET /tracks?analysis_run_id=<run-id>&class_name=person
    &observed_from_seconds=10&observed_to_seconds=20&active=true
```

스키마 v6부터 추적 결과에 침입, 선 통과, 배회 규칙을 적용할 수 있습니다.
규칙은 정규화 좌표 기반 공통 JSON으로 저장하고, 워커 시작 시 소스 이름별로
적재합니다. 발생 이벤트는 `rule_events`에 프레임·트랙과 함께 저장됩니다.

- 지원 규칙: `GET /rule-types`
- 규칙 생성·목록: `POST /rules`, `GET /rules?source_name=camera-1`
- 활성화 변경: `PATCH /rules/<rule-id>/enabled`
- 이벤트 조회: `GET /rule-events?analysis_run_id=<run-id>&event_type=intrusion_started`

구체적인 geometry와 parameters 형식 및 새 평가기 추가 방법은
[`backend/README.md`](backend/README.md)의 규칙 엔진 절을 참고합니다.

실제 RTSP 또는 Hiperwall 연동 전에
[`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)의 차단 항목을 먼저 확인합니다.
