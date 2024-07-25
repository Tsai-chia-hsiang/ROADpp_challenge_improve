import os
from pathlib import Path
import logging

class InfoFilter(logging.Filter):
    def filter(self, record):
        # Only log messages that do not have the 'console_only' attribute set to True
        return not getattr(record, 'file_only', False)


def get_logger(name: str, file: os.PathLike) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid adding them multiple times
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(InfoFilter())
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def remove_old_tf_evenfile(ckpt:Path):
    old_board_file = list(
        filter(
            lambda x:'events.out.tfevents.' in x.name,
            [_ for _ in ckpt.iterdir() if _.is_file()]
        )
    )

    for i in old_board_file:
        print(f"remove {i}")
        os.remove(i)
