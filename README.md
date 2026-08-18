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

CCTV_YOLO_ENABLED=true \
docker compose --profile rtsp run --rm --build rtsp-worker \
  --max-samples 10 --save-snapshots
```

실제 카메라를 사용할 때만 Git에서 제외된 `.env`에
`CCTV_RTSP_INPUT_URL=rtsp://username:password@camera-host:554/stream`을 설정합니다.
Hiperwall에서 원본 영상을 읽을 주소는
`rtsp://<Docker 호스트 IP>:8554/camera`입니다.

YOLO 분석 시에는 기본으로 클래스별 IoU 객체 추적이 적용되어 각 검출 결과에
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

실제 RTSP 또는 Hiperwall 연동 전에
[`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)의 차단 항목을 먼저 확인합니다.
