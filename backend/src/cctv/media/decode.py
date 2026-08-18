"""Decode, timestamp, bounded-buffer, and frame-sampling boundary.

The implementation must drop stale frames rather than allow unbounded latency
growth when downstream analysis cannot keep up.
"""
