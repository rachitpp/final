import logging
from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """One-line logger factory. Idempotent."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger
