# Media pipeline boundary

The analyzed-video restream is a long-running media pipeline. It is not a
durable queue job and must remain separate from Hiperwall display jobs.

## Data flow

```text
RTSP source
  -> ingest and reconnect
  -> decode and frame sampling
  -> detector and tracker
  -> overlay boxes, labels, and track IDs
  -> encode
  -> analyzed RTSP restream
  -> HiperSource IP Streams
```

## Module ownership

`backend/src/cctv/media/ingest.py`

- owns source connection, timeouts, reconnect policy, and source health
- does not run detection or persist events

`backend/src/cctv/media/decode.py`

- owns decode, timestamps, bounded buffering, and frame sampling
- must prevent an overloaded analyzer from growing an unbounded frame queue

`backend/src/cctv/media/overlay.py`

- owns rendering boxes, labels, track IDs, and alert state on frames
- consumes normalized analysis results and does not make alert decisions

`backend/src/cctv/media/encode.py`

- owns output resolution, codec, bitrate, GOP, and encoder lifecycle
- may select CPU or supported hardware encoding through configuration

`backend/src/cctv/media/restream.py`

- owns publication of the analyzed stream and output health
- does not call HiperInterface or control the video-wall layout

`backend/src/cctv/workers`

- owns orchestration, durable display jobs, retries, and leases
- may start or stop a media pipeline but does not contain frame processing

## Process boundary

During the MVP, the media modules remain in the backend codebase but run outside
the API request path. The API must never perform decode or inference inline.

When load testing demonstrates a need, launch the analysis pipeline as a
separate `video-worker` container without moving the module ownership above.
This makes process separation a deployment change rather than a rewrite.

## Failure rules

- Keep frame buffers bounded and prefer dropping stale frames over increasing
  end-to-end latency.
- Record source, decoder, encoder, and restream health independently.
- Reconnect with bounded exponential backoff and jitter.
- Never log RTSP credentials.
- A restream failure must not delete detection metadata already committed.
- Hiperwall display failure must not stop recording or analysis.

## Hiperwall rule

HiperInterface controls content and layout. It is not a per-frame drawing API.
If boxes and track IDs are required on the video wall, this pipeline must create
and publish a separate analyzed RTSP stream.

