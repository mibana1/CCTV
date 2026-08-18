"""Command-line entry point for registering 3-5 local face photos."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import get_settings
from cctv.db import IdentityRepository, initialize_database
from cctv.identity.face import OpenCvSFaceExtractor
from cctv.identity.registration import FaceRegistrationService

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and store SFace embeddings from 3-5 registration photos"
    )
    parser.add_argument("identity_id", help="ID returned by POST /identities")
    parser.add_argument(
        "--photo-dir",
        type=Path,
        help="photo directory; defaults to CCTV_IDENTITY_PHOTO_DIR/<external_id-or-id>",
    )
    parser.add_argument("--face-detection-model", type=Path)
    parser.add_argument("--face-embedding-model", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    configure_logging(
        level=settings.log_level,
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    try:
        initialize_database(settings.database_path)
        extractor = OpenCvSFaceExtractor(
            arguments.face_detection_model or settings.face_detection_model_path,
            arguments.face_embedding_model or settings.face_embedding_model_path,
            score_threshold=settings.face_detection_score_threshold,
            nms_threshold=settings.face_detection_nms_threshold,
            top_k=settings.face_detection_top_k,
            max_input_dimension=settings.face_detection_max_input_dimension,
        )
        result = FaceRegistrationService(
            IdentityRepository(settings.database_path),
            extractor,
            photo_root=settings.identity_photo_dir,
        ).register(arguments.identity_id, photo_directory=arguments.photo_dir)
        output = {
            "identity_id": result.identity.id,
            "external_id": result.identity.external_id,
            "display_name": result.identity.display_name,
            "photo_directory": str(result.photo_directory),
            "photo_count": len(result.items),
            "model_name": extractor.metadata.model_name,
            "model_version": extractor.metadata.model_version,
            "embedding_dimension": extractor.metadata.dimension,
            "items": [
                {
                    "photo_name": item.photo_name,
                    "detection_confidence": item.detection_confidence,
                    "embedding_id": item.embedding.id,
                    "embedding_sha256": item.embedding.embedding_sha256,
                }
                for item in result.items
            ],
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    except Exception:
        logger.exception(
            "Face registration failed",
            extra={
                "event": "face_registration_failed",
                "identity_id": arguments.identity_id,
            },
        )
        raise
    finally:
        shutdown_logging()


__all__ = ["build_argument_parser", "run"]
