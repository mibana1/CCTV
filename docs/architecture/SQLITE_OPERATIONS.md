# SQLite 운영·백업·GPU 저장소 정책

## 범위와 원칙

실행 중인 SQLite는 Backend 컨테이너 안에서만 점검하고 백업한다. Backend가 실행
중일 때 Windows에서 `runtime/cctv.db`를 SQLite 도구로 열거나 DB, `-wal`, `-shm`
파일을 직접 복사하지 않는다. 세 파일을 시점이 다른 상태로 복사하면 정상처럼 보이는
손상 백업이나 최신 트랜잭션 누락을 만들 수 있다.

Windows 도구가 꼭 필요하면 먼저 Backend를 정상 중지하고 Worker 프로세스가 모두
종료됐는지 확인한다. 운영 중 백업에는 Python `sqlite3.Connection.backup()` 또는
SQLite CLI의 `.backup`만 사용한다.

## 컨테이너 관리 명령

다음 명령은 소스 DB를 직접 복사하지 않고 온라인 snapshot을 만든 뒤 백업 DB에서
`PRAGMA integrity_check`가 정확히 `ok`인지 검증한다.

```bash
docker compose exec backend uv run --no-sync cctv-db-backup
docker compose exec backend uv run --no-sync cctv-db-check
```

기본 수동 백업 경로는 `CCTV_DATABASE_BACKUP_DIR`이며 컨테이너에서는
`/runtime/backups/manual`이다. 경로와 파일명을 지정할 수도 있다.

```bash
docker compose exec backend uv run --no-sync cctv-db-backup \
  --output /runtime/backups/manual/pre-deploy.db
docker compose exec backend uv run --no-sync cctv-db-check \
  --database /runtime/backups/manual/pre-deploy.db
```

명령 결과에는 DB, WAL, SHM 크기, `page_count`, `page_size`, `freelist_count`, schema
version과 무결성 결과가 JSON으로 출력된다. 백업 파일은 별도 복원 경로에 놓고 같은
점검 명령과 주요 행 개수 조회까지 통과해야 복원 가능 백업으로 승인한다.

## 삭제·checkpoint·VACUUM

보존 Worker의 실제 batch 삭제 후에는 `wal_checkpoint(PASSIVE)`를 수행한다. TRUNCATE
checkpoint와 VACUUM은 설정된 유지보수 창에만 허용한다. VACUUM은 분석 run, 유효한
분석 lease 또는 처리 중 Hiperwall 작업이 있으면 실행하지 않는다. 또한 freelist
비율/페이지 임계값, DB와 백업 경로의 여유 공간, 사전 온라인 백업과 전체 무결성
검사를 모두 통과해야 한다. 삭제된 페이지는 VACUUM 전에도 재사용되므로 파일 크기를
즉시 줄이는 것을 목표로 하지 않는다.

## GPU Linux 서버 저장소

GPU overlay(`compose.gpu.yaml`)는 DB/런타임, snapshot, backup을 서로 다른 Docker
volume으로 분리한다. 운영 호스트에서는 다음 조건을 적용한다.

- 실행 중 DB volume은 Linux 로컬 SSD 또는 Docker 전용 local volume에 둔다.
- `/mnt/c`, SMB, NAS, Windows 공유 드라이브에는 실행 중 SQLite를 두지 않는다.
- snapshot·clip의 대용량 쓰기는 DB volume과 분리한다.
- backup volume은 장애 도메인이 다른 디스크나 원격 백업 시스템으로 내보낸다. 기본
  named volume을 그대로 쓰면 논리적으로만 분리될 수 있으므로 실제 mount/driver를
  배포 환경에서 별도 디스크로 지정한다.
- DB volume 용량, 쓰기 지연, `cctv.db-wal` 크기, `freelist_count`와 backup 성공 여부를
  모니터링한다.

GPU 서버 전환 전에 현재 bind mount의 DB를 실행 중 복사하지 않는다. 기존 Backend를
중지하거나 온라인 백업을 만든 뒤, 검증된 백업을 새 local volume에 복원하고
`cctv-db-check`를 통과시킨 다음 Backend를 시작한다.
