# PostgreSQL 전환 검토 기준과 목표 구조

- 상태: 검토 완료, 전환 미결정
- 관련 미정 사항: [OQ-012](../OPEN_QUESTIONS.md)
- 현재 결정: SQLite WAL을 유지하며 근거가 수집될 때까지 PostgreSQL 구현을 시작하지 않음

## 범위

이 문서는 PostgreSQL 전환 코드나 `DATABASE_URL` 호환 계층을 구현하지 않는다.
현재 SQLite 병목을 측정하는 기준, 전환이 결정됐을 때 필요한 구조와 이행 순서만
정의한다. 전환은 아래 기준의 운영 또는 부하 시험 근거를 확보한 뒤 별도 ADR과
작업으로 진행한다.

## 현재 SQLite 직접 의존성

PostgreSQL 전환은 환경변수나 드라이버 하나를 교체하는 작업이 아니다.

- `cctv.db.database`가 `sqlite3.Connection`, `sqlite3.Row`, `?` placeholder,
  `PRAGMA foreign_keys`, `busy_timeout`, WAL, `user_version`을 직접 사용한다.
- Repository가 `sqlite3.Row`, `lastrowid`, SQLite 예외 타입과 트랜잭션 동작에
  의존한다. API와 provisioning 코드도 `sqlite3.IntegrityError` 등을 직접 처리한다.
- v0001~v0020 migration은 `sqlite3.Connection`을 받고 SQLite DDL과
  `PRAGMA table_info`를 사용한다. 이 migration을 PostgreSQL에 그대로 실행할 수 없다.
- 분석 lease, maintenance lease, Hiperwall 작업 claim은 현재 SQLite의 단일 writer와
  upsert 동작을 전제로 한다.
- 보존 Worker의 `page_count`, `freelist_count`, WAL checkpoint, 온라인 backup,
  VACUUM 정책은 SQLite 전용 유지보수 기능이다.
- TEXT timestamp, JSON 문자열, 임베딩 BLOB, 자동 증가 ID의 PostgreSQL 타입과
  반환 방식도 별도로 정해야 한다.

따라서 아직 PostgreSQL 구현이 없는 상태에서 `CCTV_DATABASE_URL` 같은 설정만
추가하지 않는다. 지원하지 않는 선택지가 존재하는 것처럼 보이면 운영자가 안전하지
않은 전환을 시도할 수 있다.

## 전환 판단 기준

아래 수치는 초기 관찰용 임시 기준이다. 실제 카메라 수, 분석 FPS, GPU 처리량과
운영 SLO가 확정되면 같은 항목의 값만 교체한다.

### 즉시 전환 검토를 시작하는 조건

다음 중 하나라도 확인되면 OQ-012의 근거를 첨부하고 PostgreSQL ADR을 작성한다.

- 쓰기 가능한 Backend 인스턴스를 2개 이상 동시에 운영해야 한다.
- `database is locked` 또는 이에 해당하는 lock timeout 때문에 프레임, 규칙 이벤트,
  Hiperwall 작업 저장이 한 번이라도 유실되거나 운영 장애가 발생한다.
- 보존 batch 삭제 또는 SQLite 유지보수를 수행하려면 실시간 분석을 중단해야 한다.
- 검증된 SQLite 최대 쓰기 처리량보다 목표 피크 쓰기량이 크다.

### 반복 측정으로 판단하는 성능 조건

다음 중 두 항목 이상이 동일 부하에서 3회 반복되면 전환 후보로 판단한다.

| 관찰 항목 | 임시 기준 | 측정 범위 |
|---|---:|---|
| DB lock 오류 | 15분 부하 시험 중 1건 이상 | 모든 API·Worker DB 작업 |
| 프레임 저장 transaction p95 | 100ms 초과 또는 분석 프레임 간격의 25% 초과 | 프레임·검출·추적·이벤트·작업 원자 저장 |
| Hiperwall claim/완료 transaction p95 | 100ms 초과 | pending claim, retry, success/failure 기록 |
| API DB 조회 p95 | 유휴 상태의 2배 초과 또는 300ms 초과 | 분석 실행·검출·규칙 이벤트 목록 |
| 보존 작업 영향 | 보존 실행 중 저장 p95가 평시의 2배 초과 | 동일 batch/카메라 부하 비교 |
| 쓰기 처리량 여유 | 검증 포화 처리량이 목표 피크의 2배 미만 | GPU 목표 FPS와 이벤트 비율 포함 |

단발성 개발 PC 결과만으로 전환하지 않는다. 운영과 같은 스토리지, 동시 카메라 수,
분석 FPS, Backend 수, API 조회 부하를 사용해야 한다.

## 먼저 확보할 관측 자료

PostgreSQL 구현 전에 다음 값을 구조화 로그 또는 metrics로 측정할 수 있어야 한다.
이 항목의 계측 자체는 별도 작업이며 이 문서에서는 구현하지 않는다.

- DB operation별 transaction 시간과 성공/실패 건수
- lock/busy timeout과 재시도 횟수 및 최종 실패 operation
- 프레임 저장 요청 시각부터 commit 완료까지의 지연
- Hiperwall pending 작업 수, claim 지연, processing 체류 시간
- API endpoint별 DB 조회 시간과 반환 행 수
- 보존 Worker 실행 전후 저장 지연, 삭제 batch 시간, checkpoint 결과
- 카메라 수·분석 FPS·검출 수에 따른 초당 insert/update/delete 수

부하 시험 결과에는 최소한 p50/p95/p99, 최대값, 오류 수, 시험 시간, DB/WAL 크기와
호스트 디스크 정보를 함께 남긴다.

## 목표 구조

전환이 결정되면 API와 Worker가 데이터베이스 드라이버를 직접 알지 않도록 다음
경계를 만든다.

```text
API / Analysis Worker / Hiperwall Worker / Retention Worker
                         |
                 Repository contracts
                         |
                transaction / unit of work
                    /                 \
          SQLite adapter          PostgreSQL adapter
          SQLite migration        PostgreSQL migration
          WAL maintenance         server maintenance policy
```

### 공통 경계

- 도메인 record와 Repository 공개 메서드는 데이터베이스 Row 타입을 노출하지 않는다.
- 프레임·검출·추적·규칙 이벤트·display action의 원자 저장 경계를 공통 contract로
  유지한다.
- connection 생성, placeholder, row mapping, 무결성 오류 변환을 adapter 안으로
  이동한다.
- API는 `sqlite3.IntegrityError` 대신 애플리케이션의 공통 conflict/not-found 오류를
  처리한다.
- 페이지 조회 순서, nullable 값, UTC timestamp와 transaction rollback 동작을
  contract test로 고정한다.
- sync/async 드라이버 선택은 실제 API 동시성과 Worker 배치 benchmark 후 결정한다.
  전환 전부터 ORM이나 async 계층을 선제 도입하지 않는다.

### PostgreSQL에서 다시 구현할 의미론

- 자동 증가 키는 `RETURNING`으로 가져오고 `lastrowid`를 사용하지 않는다.
- 시간은 `timestamptz`와 UTC를 사용한다.
- JSON payload는 `jsonb`, 임베딩 원본 호환 저장은 우선 `bytea`를 검토한다.
  벡터 검색 요구가 확인되기 전에는 pgvector를 전환 범위에 포함하지 않는다.
- Hiperwall 작업 claim은 여러 Backend가 안전하게 경쟁하도록
  `FOR UPDATE SKIP LOCKED` 등의 PostgreSQL transaction으로 다시 검증한다.
- lease는 DB 서버 시각과 원자 upsert를 사용하고 만료·소유권 contract를 유지한다.
- 보존 삭제는 PostgreSQL에 없는 `DELETE ... LIMIT`을 가정하지 않고, 제한된 ID를
  CTE로 선택한 뒤 batch 삭제한다.
- SQLite WAL checkpoint와 freelist 기반 VACUUM을 PostgreSQL에 복제하지 않는다.
  PostgreSQL에서는 autovacuum/ANALYZE 상태, dead tuple, table bloat와 운영 백업
  정책을 별도로 사용한다.

### Migration 경계

기존 SQLite migration은 수정하거나 PostgreSQL에서 재사용하지 않는다.
PostgreSQL용 baseline schema와 이후 migration chain을 별도로 만든다. 도구는
PostgreSQL 전환 ADR에서 결정하되 다음 조건을 만족해야 한다.

- 기존 FK, cascade/restrict, unique/check constraint와 인덱스를 명시적으로 대응한다.
- 이벤트 프레임과 Hiperwall 감사 이력의 생명주기를 보존한다.
- 등록 인물과 임베딩을 자동 보존 삭제 대상에서 제외한다.
- forward migration, 빈 DB 설치, 실제 크기 복제본 upgrade를 모두 검증한다.
- SQLite와 PostgreSQL migration version을 하나의 숫자로 혼동하지 않는다.

## 단계별 이행 순서

1. **관측과 benchmark**: 현재 SQLite의 기준 부하와 전환 trigger를 측정한다.
2. **ADR 작성**: PostgreSQL 필요성, 드라이버, migration 도구, 운영 주체, 백업과
   복구 목표를 결정하고 OQ-012를 `DECIDED`로 변경한다.
3. **Repository contract 분리**: 현재 SQLite 동작을 바꾸지 않은 채 transaction과
   오류 contract test를 먼저 만든다.
4. **PostgreSQL adapter/schema 구현**: 별도 개발·CI 환경에서 같은 contract를
   통과시킨다. 이때만 PostgreSQL 연결 설정을 공개한다.
5. **데이터 이행 리허설**: 운영 복제본으로 행 수, FK, checksum 가능한 payload,
   최근 이벤트와 identity embedding을 검증하고 예상 중단 시간을 측정한다.
6. **부하·장애 시험**: 목표의 2배 피크, Backend 다중 실행, Worker 강제 종료,
   Hiperwall retry, 보존 batch가 함께 동작하는지 확인한다.
7. **전환과 rollback 준비**: 쓰기를 정지하고 최종 증분/전체 복사, 검증, 연결 전환을
   수행한다. SQLite 원본은 rollback 기간 동안 읽기 전용으로 보관한다.

초기 전환에서는 dual-write를 기본 해법으로 사용하지 않는다. 두 DB 사이의 원자성을
보장할 추가 구조가 없으면 오히려 이벤트와 Hiperwall 작업 불일치를 만들 수 있다.

## 전환 완료 조건

다음을 모두 충족해야 PostgreSQL 전환을 완료로 표시한다.

- Repository contract test가 SQLite와 PostgreSQL에서 모두 통과한다.
- 프레임·이벤트·display action 원자 저장과 rollback이 동일하다.
- 여러 Backend에서 lease와 Hiperwall 작업이 중복 처리되지 않는다.
- 보존 삭제가 이벤트/Hiperwall 감사 이력과 등록 인물 데이터를 보호한다.
- 데이터 이행 전후 핵심 테이블 행 수, FK 검사와 표본 payload가 일치한다.
- 목표 피크의 2배 부하에서 정한 latency/error SLO를 만족한다.
- 백업 복구 시험과 SQLite rollback 절차가 성공한다.

## 현재 결론

현재는 PostgreSQL 필요성을 입증한 benchmark나 lock 장애 근거가 없다. 따라서
SQLite WAL, lease, batch retention 정책을 유지한다. 위 trigger가 확인되면
PostgreSQL을 단순 설정 변경이 아닌 별도 아키텍처·데이터 이행 프로젝트로 진행한다.
