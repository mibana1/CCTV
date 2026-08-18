# Model artifacts

This directory is the local mount point for model binaries such as `.pt`,
`.onnx`, `.engine`, and `.torchscript` files. Model binaries are deliberately
excluded from Git.

## Required metadata

Each deployed model must have the following information recorded in deployment
configuration or release notes:

- model name and upstream source
- model version
- file name
- SHA-256 checksum
- supported object classes
- input size
- runtime (`pytorch`, `onnxruntime`, or `tensorrt`)
- license and commercial-use review status

## Environment variables

Use configuration instead of hard-coded paths:

```text
CCTV_MODEL_PATH=/models/model.onnx
CCTV_MODEL_CLASSES_PATH=/models/classes.txt
CCTV_AI_DEVICE=cpu
CCTV_FACE_DETECTION_MODEL_PATH=/models/face/face_detection_yunet_2026may.onnx
CCTV_FACE_EMBEDDING_MODEL_PATH=/models/face/face_recognition_sface_2021dec.onnx
```

현재 CPU 워커는 OpenCV DNN에서 읽을 수 있는 ONNX 모델을 사용하며, 임베디드
NMS 없이 export된 일반 YOLOv8/YOLO11 또는 YOLOv5 출력을 지원합니다. COCO
80개 클래스 모델은 클래스 파일을 생략할 수 있습니다. 커스텀 모델은 모델의
출력 클래스 순서와 정확히 일치하도록 UTF-8 텍스트 파일에 한 줄당 하나의
클래스 이름을 작성합니다.

## Container mount

Mount this directory read-only in the analysis container:

```yaml
volumes:
  - ./artifacts/models:/models:ro
```

Never download a model silently during normal application startup. Model
acquisition must be an explicit setup or deployment step, followed by checksum
verification.

## Face registration models

Face registration uses the official OpenCV Zoo ONNX files below. Keep them in
the `face/` subdirectory, which is also excluded from Git.

| Role | File | Version | SHA-256 | License |
| --- | --- | --- | --- | --- |
| detection/alignment | `face_detection_yunet_2026may.onnx` | YuNet 2026may | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` | MIT |
| embedding | `face_recognition_sface_2021dec.onnx` | SFace 2021dec | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

Record the verified SHA-256 values of deployed files in release or deployment
metadata. Do not substitute an identically named file without re-verification.
