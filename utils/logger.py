"""
Centralized logging setup using loguru.
Import `logger` from here everywhere in the project.
"""

import sys
from loguru import logger
from core.config import get_settings


def setup_logger() -> None:
    """Configure loguru with the level from settings."""
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/rag_doctor.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


setup_logger()

__all__ = ["logger"]
