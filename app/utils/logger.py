# Configuration for structured logging across the application.
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Generator

import structlog


def _configure_structlog() -> None:
    env = os.getenv("APP_ENV", "development").lower()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    renderer = structlog.processors.JSONRenderer() if env == "production" \
        else structlog.dev.ConsoleRenderer(colors=False)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors[:6],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_structlog()


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def log_node_execution(node_name: str, trace_id: str) -> Generator[None, None, None]:
    log = get_logger("node_execution")
    start = time.perf_counter()
    log.info("node_start", node=node_name, trace_id=trace_id)
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log.info("node_end", node=node_name, trace_id=trace_id,
                 duration_ms=round(duration_ms, 2))
