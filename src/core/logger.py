"""
logger.py

EFP 共通ログ管理モジュール
"""

from pathlib import Path
import logging

from src.core.config import LOG_DIR

# ログフォルダ作成
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "efp.log"


def get_logger(name: str) -> logging.Logger:
    """
    共通Loggerを取得する。

    Args:
        name: Logger名

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ファイル出力
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    # コンソール出力
    stream_handler = logging.StreamHandler()

    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(stream_handler)

    return logger