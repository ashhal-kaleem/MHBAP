"""
Centralised logging configuration using Loguru.
Provides structured JSON logs in production, coloured console in dev.
"""
import sys
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """Configure Loguru for the application environment."""
    logger.remove()  # remove default handler

    if settings.APP_ENV == "production":
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DDTHH:mm:ss.sssZ} | {level} | {name}:{line} | {message}",
            level="INFO",
            serialize=True,  # JSON output
        )
    else:
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )

    logger.info(
        "Logging configured",
        env=settings.APP_ENV,
        debug=settings.DEBUG,
    )
