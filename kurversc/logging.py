"""Small, opt-in structlog configuration for KurveRSC progress output."""

from __future__ import annotations

import logging

import structlog


def configure_logging(*, level: int = logging.INFO, colors: bool = True) -> None:
    """Configure concise console rendering for KurveRSC progress events."""

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=colors),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
