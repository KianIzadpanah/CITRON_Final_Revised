import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def guard_overwrite(path: str | Path, force: bool) -> bool:
    """Return True if it is safe to write (file absent or force=True)."""
    p = Path(path)
    if p.exists() and not force:
        logging.getLogger("io_utils").warning(
            f"Output already exists, skipping: {p}  (use --force to overwrite)"
        )
        return False
    return True


def log_failure(scene_id: str, reason: str, failure_log: str | Path) -> None:
    Path(failure_log).parent.mkdir(parents=True, exist_ok=True)
    with open(failure_log, "a") as f:
        f.write(f"{scene_id}\t{reason}\n")
