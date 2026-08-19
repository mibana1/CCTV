# MediaMTX

`mediamtx.yml`은 원본 카메라 중계 경로 `camera`와 향후 오버레이 영상을 위한
발행 경로 `analyzed`를 정의합니다. Docker Compose는 MediaMTX `1.19.3`을
고정해서 사용합니다.

- `CCTV_RTSP_INPUT_URL`이 비어 있으면 `camera`는 로컬 테스트 발행자를 받습니다.
- 값이 있으면 MediaMTX가 해당 RTSP 카메라를 한 번만 pull합니다.
- 분석 워커는 컨테이너 내부의 `rtsp://mediamtx:8554/camera`를 읽습니다.
- Hiperwall은 `rtsp://<Docker 호스트 IP>:8554/camera`를 읽을 수 있습니다.
- 기본 `camera` 경로의 계정 정보는 `.env`에만 저장하고 Git에 추가하지 않습니다.
- 대시보드에서 추가한 카메라는 `cam-<고유값>` 경로로 Control API에 등록됩니다.
- Control API는 `127.0.0.1:9997`에만 열리고 내부 프록시는 Backend 주소만 허용합니다.
- 동적 카메라 자격 증명은 SQLite에 암호화되고 키는 `runtime/secrets`에 보관됩니다.

로컬 영상으로 중계를 확인할 때는 저장소 루트에서 다음을 실행합니다.

```bash
CCTV_RTSP_INPUT_URL=publisher \
docker compose --profile rtsp-test up -d mediamtx rtsp-test-publisher
```

분석 워커를 10개 샘플로 제한해 확인하려면 다음을 실행합니다.

```bash
CCTV_DETECTOR_ENABLED=true \
docker compose --profile rtsp run --rm --build rtsp-worker --max-samples 10 --save-snapshots
```

`analyzed` 경로는 현재 예약만 되어 있습니다. 박스 오버레이·인코딩 구현 후
별도 publisher가 이 경로에 분석 영상을 발행합니다.
