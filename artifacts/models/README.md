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
MODEL_PATH=/models/yolo-model.pt
MODEL_DEVICE=cpu
MODEL_VERSION=unassigned
MODEL_SHA256=unassigned
```

The development default is `MODEL_DEVICE=cpu`. CUDA is enabled only on a test
machine that has a supported NVIDIA GPU and driver.

## Container mount

Mount this directory read-only in the analysis container:

```yaml
volumes:
  - ./artifacts/models:/models:ro
```

Never download a model silently during normal application startup. Model
acquisition must be an explicit setup or deployment step, followed by checksum
verification.
