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
cd "/mnt/c/Users/노주형/Desktop/CCTVTest/CCTV/backend"
uv sync
uv run pytest
uv run cctv
```

API 확인 주소는 `http://127.0.0.1:8000/health`입니다.

## Docker 시작

저장소 루트에서 다음을 실행합니다.

```bash
cp .env.example .env
docker compose up --build
```

실제 RTSP 또는 Hiperwall 연동 전에
[`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)의 차단 항목을 먼저 확인합니다.

