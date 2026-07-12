"""structlog configuration. Trimmed from PCC's: conductor has no request
middleware yet (the HTTP conversations API arrives in Slice 4), so this is just
the one-shot ``configure_logging`` — needed now for the MCP server, whose stdout
is the JSON-RPC transport and must never carry a log line.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog

from app.config import get_settings

__all__ = ["configure_logging"]


def configure_logging(stream: TextIO | None = None) -> None:
    """Configure structlog + stdlib logging to ``stream`` (default stdout).

    The MCP server passes ``sys.stderr``: its stdout carries the JSON-RPC
    protocol, so a single log line on stdout would corrupt the transport.
    JSON in production, human-readable console output otherwise.
    """
    settings = get_settings()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    renderer: structlog.types.Processor
    if settings.app_env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
