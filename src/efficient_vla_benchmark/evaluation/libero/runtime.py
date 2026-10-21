"""Import the pinned simulator without writing a shared, cross-user log file."""

import importlib
import logging
import threading
from pathlib import Path

_IMPORT_LOCK = threading.RLock()


def load_environment(log_dir: Path):
    """Redirect only robosuite's legacy default during its first, serial import.

    The evaluation CLI owns one simulator per process. No installed package or
    global factory remains patched after import, including on import failure.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    with _IMPORT_LOCK:
        original = logging.FileHandler

        class RunFileHandler(original):
            def __init__(self, filename, *args, **kwargs):
                if Path(filename) == Path("/tmp/robosuite.log"):
                    filename = log_dir / "robosuite.log"
                super().__init__(filename, *args, **kwargs)

        logging.FileHandler = RunFileHandler
        try:
            return importlib.import_module("libero.libero.envs").OffScreenRenderEnv
        finally:
            logging.FileHandler = original
