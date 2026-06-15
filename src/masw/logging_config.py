import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(processName)s | %(name)s | %(levelname)s | %(message)s",
    )
