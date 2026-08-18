"""Long-running CCTV media pipeline boundaries.

Frame ingestion, decoding, overlay rendering, encoding, and RTSP publication
belong in this package. Durable job retries and Hiperwall layout operations do
not.

See ``docs/architecture/MEDIA_PIPELINE.md`` before adding implementations.
"""
