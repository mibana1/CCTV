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

로컬 디코더는 API와 별도인 단일 실행 워커에서 구동합니다. 입력 경로는 명령행
인자 또는 `CCTV_LOCAL_VIDEO_PATH`로 지정하고, 개발 중에는 처리할 샘플 수를
제한할 수 있습니다. 코어 워커의 기본 소비자는 프레임을 저장하지 않지만 CLI에서
`--save-snapshots`를 사용하면 각 실행의 고유 하위 디렉터리에 JPEG를 원자적으로
저장합니다. 파일명에는 샘플 번호, 원본 프레임 번호와 타임스탬프가 포함되며
기존 실행 결과를 덮어쓰지 않습니다.

분석 워커는 구체 모델 대신 `ObjectDetector` 인터페이스와 `DetectorFactory`에
의존합니다. `BoundingBox`, `Detection`, `FrameDetections`는 모델 독립적인 공통
결과 계약입니다. 기본 `CCTV_DETECTOR_TYPE=yolo_onnx`이며
`CCTV_DETECTOR_ENABLED=true` 또는 `--analyze`로 분석을 활성화합니다. 기존
`CCTV_YOLO_ENABLED`와 `--analyze-yolo`도 호환을 위해 계속 지원합니다.

새 객체 검출 모델을 추가할 때는 `ObjectDetector`의 `metadata`, `summary`,
`analyze()`를 구현하고 `DetectorFactory`에 고유 타입과 builder를 등록합니다.
워커, 추적기, 규칙 엔진, DB 저장 코드는 변경하지 않습니다. 모델별 추가 옵션은
`DetectorConfig.options`로 전달할 수 있습니다. DB에는 실행별 `detector_type`도
저장되어 모델 교체 후 결과를 구분할 수 있습니다.

기본 YOLO 분석기는 OpenCV DNN으로 ONNX 모델을 로드하며 모델을 자동으로
다운로드하지 않습니다. `--analyze` 또는 `CCTV_DETECTOR_ENABLED=true`로
명시적으로 활성화하고 `CCTV_MODEL_PATH`에 로컬 모델을 배치합니다. 표준 COCO
80개 클래스는 내장 이름을 사용하며, 커스텀 모델은 `CCTV_MODEL_CLASSES_PATH`에
출력 순서와 일치하는 클래스 이름 파일을 지정합니다. YOLOv8/YOLO11의
`4 + classes` 출력과 YOLOv5의 `5 + classes` 출력을 지원하고 클래스별 NMS 후
박스를 원본 프레임 좌표로 복원합니다. 프레임별 결과는 DEBUG 구조화 로그에,
전체 처리량과 평균 추론 시간은 종료 JSON 및 `detector_run_completed` 로그에
기록됩니다.

YOLO가 활성화되면 `CCTV_PERSIST_DETECTIONS=true` 기본값에 따라 실행 정보,
분석한 프레임과 검출 객체를 `CCTV_DATABASE_PATH`의 SQLite에 저장합니다. 각
실행은 UUID로 구분되며 모델 파일명과 SHA-256, 임계값, FPS, 완료 상태를 함께
기록합니다. 검출 객체가 없는 프레임도 저장하므로 처리 프레임 수와 검출 수를
독립적으로 확인할 수 있습니다. 프레임과 그 검출 결과는 하나의 트랜잭션으로
저장되며 실행 중 오류가 발생하면 실행 상태를 `failed`로 종료합니다.

```bash
uv run cctv-local-worker ../video/testvideo1.mp4 --sample-fps 2 --max-samples 10 --save-snapshots
uv run cctv-local-worker ../video/testvideo1.mp4 --max-samples 10 --analyze --model-path ../artifacts/models/model.onnx
docker compose --profile local-video run --rm --build local-video-worker
```

Compose에서 CPU 분석을 켜려면 모델 파일을 `artifacts/models/model.onnx`에 두고
`CCTV_DETECTOR_ENABLED=true`를 설정합니다. 파일명이 다르면 `YOLO_MODEL_FILE`로
마운트 디렉터리 안의 파일명만 변경할 수 있습니다. 커스텀 클래스 파일은 컨테이너
경로(예: `/models/classes.txt`)를 `CCTV_MODEL_CLASSES_PATH`로 지정합니다.

검출 조회 API는 내부 영상·모델 파일 경로를 반환하지 않습니다. 모든 목록은
`page`와 `limit`을 사용하며 `limit`은 최대 100입니다.

```text
GET /analysis-runs?page=1&limit=50
GET /analysis-runs/{analysis_run_id}
GET /detections?analysis_run_id={id}&class_name=person&min_confidence=0.5&page=1&limit=50
```

`/analysis-runs`는 `status=completed|stopped|failed|running` 필터를 지원하고,
`/detections`의 클래스 이름은 대소문자를 구분하지 않는 완전 일치 방식입니다.
현재 API에는 사용자 인증이 없으므로 로컬 검증 네트워크 밖에 공개하지 마십시오.

## 인물 등록 및 임베딩

인물 프로필과 얼굴 임베딩은 분리해 저장합니다. 한 인물에 여러 등록 샘플과 여러
모델 버전의 임베딩을 연결할 수 있으며, 모델명·버전·차원이 일치하는 벡터만 향후
매칭 후보로 불러올 수 있습니다. 입력 벡터는 유한값과 차원(2..4096)을 검증한 뒤
L2 정규화된 little-endian float32 BLOB으로 저장하고 SHA-256을 함께 기록합니다.
API 응답에는 원본 벡터를 반환하지 않습니다.

```text
POST   /identities
GET    /identities?query=EMP-001&enabled=true&page=1&limit=50
GET    /identities/{identity_id}
PATCH  /identities/{identity_id}
DELETE /identities/{identity_id}

POST   /identities/{identity_id}/embeddings
GET    /identities/{identity_id}/embeddings?model_name=arcface&model_version=1.0
DELETE /identities/{identity_id}/embeddings/{embedding_id}
```

```json
POST /identities/{identity_id}/embeddings
{
  "model_name": "arcface",
  "model_version": "1.0",
  "vector": [0.12, -0.34, 0.56],
  "source_reference": "registration-image-1",
  "quality_score": 0.93
}
```

등록 사진은 `CCTV_IDENTITY_PHOTO_DIR/<external_id-or-identity-id>` 아래에 JPG 또는
PNG 형식으로 3~5장을 둡니다. `cctv-register-faces`는 모든 사진에서 정확히 한 명의
얼굴을 YuNet으로 검출하고 5개 랜드마크로 정렬한 뒤 SFace 2021dec의 128차원
임베딩을 생성합니다. 사진 하나라도 읽을 수 없거나 얼굴이 없거나 두 명 이상이면
아무 임베딩도 저장하지 않습니다. 모든 추출이 성공한 경우에만 한 트랜잭션으로
저장하며 사진 원본과 임베딩 벡터는 출력하지 않습니다.

```text
runtime/identity_images/EMP-001/
├── 01_front.jpg
├── 02_left.jpg
└── 03_right.jpg
```

```bash
# 로컬 개발 환경
cd backend
uv run cctv-register-faces {identity_id}

# Compose의 /runtime, /models 마운트 사용
docker compose run --rm backend \
  uv run --no-sync cctv-register-faces {identity_id}
```

기본 모델 파일은 각각
`artifacts/models/face/face_detection_yunet_2026may.onnx`와
`artifacts/models/face/face_recognition_sface_2021dec.onnx`에 명시적으로 설치해야
하며 애플리케이션이 실행 중 자동 다운로드하지 않습니다. 고해상도 입력은 긴 변을
기본 960px로 축소해 검출한 뒤 얼굴 좌표를 원본 프레임 기준으로 복원합니다.

로컬 영상과 RTSP 워커의 `--match-faces`는 프레임 안의 모든 얼굴을 검출하고,
등록 사진별 코사인 유사도 중 인물별 최고값을 비교합니다. 최고 점수가 기본 0.45
이상이고 두 번째 인물과의 차이가 기본 0.05 이상일 때만 등록 인물로 판정합니다.
결과 JSON과 구조화 로그에는 점수·판정·얼굴 좌표를 기록하지만 임베딩 벡터는
기록하지 않습니다.

```bash
# 로컬 영상 얼굴 비교
uv run cctv-local-worker ../video/testvideo1.mp4 \
  --match-faces --sample-fps 2 --max-samples 30

# MediaMTX 카메라 얼굴 비교
docker compose --profile rtsp run --rm --build rtsp-worker \
  --match-faces --sample-fps 2 --max-samples 120
```

환경변수 `CCTV_FACE_MATCHING_ENABLED=true`로 항상 켤 수도 있고,
`CCTV_FACE_MATCH_SIMILARITY_THRESHOLD`와 `CCTV_FACE_MATCH_MINIMUM_MARGIN`으로
판정 기준을 조정할 수 있습니다. 임베딩은 생체정보이므로 실제 운영에서는 API
인증·권한, 전송 암호화, SQLite 파일 암호화 또는 접근 통제와 보존·삭제 정책을
추가해야 합니다.

## 규칙 엔진

`CCTV_RULES_ENABLED=true`(기본값), 객체 추적, 검출 저장이 모두 활성화된 분석
워커는 시작할 때 해당 `source_name`의 활성 규칙을 SQLite에서 읽고 검증합니다.
좌표는 해상도와 무관한 `0.0..1.0` 범위이며 객체 박스의 하단 중앙점을 판정
기준으로 사용합니다. 규칙을 API로 변경한 경우 실행 중인 워커에는 영향을 주지
않으며 다음 워커 시작부터 적용됩니다.

- `intrusion`: polygon 진입 시 `intrusion_started`, 이탈 시 `intrusion_ended`
- `line_crossing`: 유한 line 통과 시 방향을 포함한 `line_crossed`
- `loitering`: polygon 안에서 지정 시간 체류 시 `loitering_started`, 이탈 시
  `loitering_ended`

```json
POST /rules
{
  "name": "출입 금지 구역",
  "source_name": "camera-1",
  "rule_type": "intrusion",
  "class_name": "person",
  "geometry": {
    "type": "polygon",
    "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
  },
  "parameters": {
    "dwell_seconds": 0,
    "missing_tolerance_seconds": 2,
    "cooldown_seconds": 5
  }
}
```

`line_crossing`은 `geometry.type=line`과 두 점을 사용하며 `direction`은 `any`,
`negative_to_positive`, `positive_to_negative` 중 하나입니다. `loitering`은
`duration_seconds`가 기본 30초입니다. `GET /rule-types`, `GET /rules`,
`PATCH /rules/{id}/enabled`, `GET /rule-events`로 지원 유형·설정·발생 이력을
조회하거나 활성 상태를 바꿀 수 있습니다. 이벤트는 해당 프레임, 검출, 트랙과
같은 SQLite 트랜잭션으로 저장됩니다.

새 규칙은 `RuleEvaluator` 프로토콜에 맞는 독립 평가기를 만들고
`built_in_evaluators()`에 등록합니다. 엔진, 워커, DB 스키마는 규칙 유형을
하드코딩하지 않으므로 고유 `rule_type`, 설정 검증, `evaluate()`만 추가하면
됩니다. 현재 체류·쿨다운 상태는 워커 메모리에 있으므로 워커 재시작 시
초기화됩니다.

## RTSP 및 MediaMTX

MediaMTX는 외부 카메라 RTSP를 `camera` 경로로 중계합니다. RTSP 워커는
API와 별도 프로세스로 이 내부 경로를 읽고, 경과 시간 기준 `CCTV_ANALYSIS_FPS`만
YOLO와 저장 계층에 전달합니다. OpenCV 열기·읽기 타임아웃을 적용하고 연결이
끊어지면 jitter가 포함된 제한적 지수 백오프로 재연결합니다. Python 측 프레임
프리페치 큐는 만들지 않습니다.

```bash
# 로컬 테스트 영상 -> MediaMTX camera 경로
CCTV_RTSP_INPUT_URL=publisher \
docker compose --profile rtsp-test up -d mediamtx rtsp-test-publisher

# MediaMTX camera 경로 -> YOLO -> SQLite (10개 샘플 후 종료)
CCTV_DETECTOR_ENABLED=true \
docker compose --profile rtsp run --rm --build rtsp-worker \
  --max-samples 10 --save-snapshots
```

실제 카메라에서는 Git에서 제외된 `.env`에 다음처럼 입력한 뒤 서비스를
시작합니다. 사용자명이나 비밀번호에 `@`, `:`, `/`, `#` 같은 문자가 있으면
URL percent-encoding을 적용해야 합니다.

```dotenv
CCTV_RTSP_INPUT_URL=rtsp://username:password@camera-host:554/stream
CCTV_RTSP_SOURCE_NAME=camera-1
CCTV_DETECTOR_ENABLED=true
```

```bash
docker compose --profile rtsp up -d --build mediamtx rtsp-worker
```

워커와 DB에는 RTSP URL 대신 `CCTV_RTSP_SOURCE_NAME`만 기록합니다. Hiperwall의
IP Stream 주소는 `rtsp://<Docker 호스트 IP>:8554/camera`입니다. 이 경로는
현재 원본 영상이며 `analyzed` 경로는 박스 오버레이·인코딩 구현 후 사용합니다.
