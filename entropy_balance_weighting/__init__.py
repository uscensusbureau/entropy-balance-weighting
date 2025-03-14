import logging
import os

ncpus = str(int(os.getenv("NCPUS", "3")) - 2)
os.environ.setdefault("POLARS_MAX_THREADS", ncpus)
os.environ.setdefault("OMP_NUM_THREADS", ncpus)
os.environ.setdefault("NUMEXPR_NUM_THREADS", ncpus)
os.environ.setdefault("MKL_NUM_THREADS", ncpus)
os.environ.setdefault("OPENBLAS_NUM_THREADS", ncpus)

from entropy_balance_weighting.ebw_penalty import entropy_balance_penalty  # noqa: E402
from entropy_balance_weighting.ebw_routines import entropy_balance  # noqa: E402

__all__ = ["entropy_balance", "entropy_balance_penalty", "setup_logging"]


def setup_logging(filepath: str, mode: str = "w") -> logging.Logger:
    """Get the logger pointed to a file."""
    logger = logging.getLogger("entropy_balance_weighting.ebw_routines")
    logger.setLevel(logging.INFO)

    filelog = logging.FileHandler(filepath, mode=mode)
    filelog.setLevel(logging.INFO)
    logger.info("entropy_balance_weighting.ebw_routines")
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    filelog.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(filelog)
    return logger
