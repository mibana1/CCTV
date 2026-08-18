# Open questions and external confirmations

This file is the single source of truth for unresolved project questions.
Do not duplicate open-question lists in other documents. Other documents should
link here.

Status values:

- `OPEN`: confirmation or a decision is still required.
- `IN_PROGRESS`: evidence is being collected.
- `BLOCKED`: progress requires external access or another party.
- `DECIDED`: a decision record exists under `docs/decisions/`.

## Blocking before real integration

| ID | Question or confirmation | Evidence required | Owner | Status | Decision |
|---|---|---|---|---|---|
| OQ-001 | Do the DVR/encoders provide RTSP and ONVIF? | Model, firmware, vendor documentation, and a successful connection test | Unassigned | OPEN | - |
| OQ-002 | Is an analysis substream available per camera? | Codec, resolution, FPS, bitrate, and RTSP URL pattern | Unassigned | OPEN | - |
| OQ-003 | What is the DVR's maximum simultaneous RTSP connection count? | Vendor limit and measured reconnect behavior | Unassigned | OPEN | - |
| OQ-004 | Is HiperInterface enabled by the installed license? | Version, build, license screen, and a successful capability query | Unassigned | OPEN | - |
| OQ-005 | Which HiperInterface authentication mode is active? | Confirm `None`, `Token`, or `Crypto` and capture a sanitized request/response | Unassigned | OPEN | - |
| OQ-006 | Will Hiperwall show the original stream or the analyzed restream? | Stakeholder decision and latency/quality comparison | Unassigned | OPEN | - |
| OQ-007 | What hardware will be used for GPU validation? | OS, GPU model, VRAM, driver, and supported CUDA runtime | Unassigned | OPEN | - |

## Required before production sizing

| ID | Question or confirmation | Evidence required | Owner | Status | Decision |
|---|---|---|---|---|---|
| OQ-008 | How long must original video, clips, and snapshots be retained? | Retention policy per data type | Unassigned | OPEN | - |
| OQ-009 | Where will original video be recorded? | NVR/RAID architecture, usable capacity, reserve, and failure policy | Unassigned | OPEN | - |
| OQ-010 | What alert latency and detection accuracy are acceptable? | Measurable target and acceptance dataset | Unassigned | OPEN | - |
| OQ-011 | How many streams must Hiperwall display simultaneously? | Licensed and measured source/display limits | Unassigned | OPEN | - |
| OQ-012 | Is PostgreSQL required for multi-worker operation? | SQLite WAL concurrency benchmark and target write rate | Unassigned | OPEN | - |

## Deferred scope decisions

| ID | Question or confirmation | Evidence required | Owner | Status | Decision |
|---|---|---|---|---|---|
| OQ-013 | Is cross-camera person re-identification required? | Stakeholder use case and privacy review | Unassigned | OPEN | - |
| OQ-014 | Are face or license-plate features permitted? | Legal, privacy, retention, and access-control review | Unassigned | OPEN | - |
| OQ-015 | Is Verkada webhook integration part of the first delivery? | Scope decision and API/webhook access | Unassigned | OPEN | - |
| OQ-016 | Which detector and tracker are approved for commercial use? | Accuracy benchmark and license review | Unassigned | OPEN | - |

## Resolution process

1. Assign an owner and collect direct evidence.
2. Update the row status and add a concise evidence link or reference.
3. For an architectural decision, create an ADR from
   `docs/decisions/000-template.md`.
4. Change the status to `DECIDED` and link the ADR in the Decision column.
5. Update implementation configuration and tests affected by the decision.

Do not mark an item decided based only on an assumption or an unverified device
specification.
