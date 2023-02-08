import logging
import sys
from pathlib import Path

from typing import Union


def set_logging(exp_name: Union[None, str, Path] = None):
    print("Set loggers" if exp_name else "Set logger")

    # disable warnings
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # set format messages
    formatter = logging.Formatter("%(message)s")
    logging.basicConfig(format="%(message)s")

    # get root logger
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # Console Handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    print("console: ", console)

    logger.addHandler(console)

    # Log File Handler
    if exp_name:
        log_path = Path('./notebooks').resolve().parents[0] / f'{exp_name}'
        log_path.mkdir(exist_ok=True, parents=True)
        file_log = logging.FileHandler(log_path / 'logs.log', mode='w')
        file_log.setFormatter(formatter)
        file_log.setLevel(logging.INFO)
        print("file: ", file_log)

        logger.addHandler(file_log)
