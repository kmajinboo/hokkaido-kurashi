"""共通APIクライアントの基底クラス。"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseAPIClient(ABC):
    """経済データAPIクライアント共通のインターフェース。"""

    @abstractmethod
    def get_series(self, series_id: str, **kwargs) -> pd.DataFrame:
        """系列を取得し、標準化したDataFrameを返す。"""
        raise NotImplementedError
