from cctv.inference import BoundingBox, Detection, FrameDetections, IoUTracker


def result(
    sample_index: int,
    timestamp_seconds: float,
    *detections: Detection,
) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index,
        sample_index=sample_index,
        timestamp_seconds=timestamp_seconds,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.01,
        detections=detections,
    )


def detection(class_id: int, box: tuple[int, int, int, int]) -> Detection:
    return Detection(
        class_id=class_id,
        label="person" if class_id == 0 else "car",
        confidence=0.9,
        box=BoundingBox(*box),
    )


def test_tracker_keeps_id_for_overlapping_detection_of_same_class() -> None:
    tracker = IoUTracker(iou_threshold=0.3)

    first = tracker.update(result(0, 0.0, detection(0, (10, 10, 110, 210))))
    second = tracker.update(result(1, 0.5, detection(0, (15, 12, 115, 212))))

    assert first.detections[0].track_id == 1
    assert second.detections[0].track_id == 1
    assert tracker.summary.processed_frames == 2
    assert tracker.summary.assigned_detections == 2
    assert tracker.summary.created_tracks == 1
    assert tracker.summary.matched_detections == 1
    assert tracker.summary.active_tracks == 1
    assert tracker.active_track_ids == (1,)


def test_tracker_never_matches_different_classes_or_two_detections_to_one_track() -> None:
    tracker = IoUTracker(iou_threshold=0.3)
    tracker.update(result(0, 0.0, detection(0, (10, 10, 110, 210))))

    tracked = tracker.update(
        result(
            1,
            0.5,
            detection(0, (12, 10, 112, 210)),
            detection(0, (14, 10, 114, 210)),
            detection(2, (10, 10, 110, 210)),
        )
    )

    ids = [item.track_id for item in tracked.detections]
    assert ids[0] == 1
    assert len(set(ids)) == 3


def test_tracker_expires_missing_and_idle_tracks() -> None:
    missing_tracker = IoUTracker(max_missed_frames=1, max_idle_seconds=10)
    missing_tracker.update(result(0, 0.0, detection(0, (10, 10, 110, 210))))
    missing_tracker.update(result(1, 0.5))
    missing_tracker.update(result(2, 1.0))
    reacquired = missing_tracker.update(result(3, 1.5, detection(0, (10, 10, 110, 210))))

    idle_tracker = IoUTracker(max_idle_seconds=1)
    idle_tracker.update(result(0, 0.0, detection(0, (10, 10, 110, 210))))
    after_gap = idle_tracker.update(result(1, 2.0, detection(0, (10, 10, 110, 210))))

    assert reacquired.detections[0].track_id == 2
    assert after_gap.detections[0].track_id == 2
    assert missing_tracker.active_track_ids == (2,)
    assert idle_tracker.active_track_ids == (2,)


def test_tracker_rejects_out_of_order_frames() -> None:
    tracker = IoUTracker()
    tracker.update(result(1, 1.0))

    try:
        tracker.update(result(1, 1.5))
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("duplicate sample index was accepted")
