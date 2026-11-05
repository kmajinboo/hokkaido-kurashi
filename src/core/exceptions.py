"""
exceptions.py

EFP 共通例外クラス
"""


class EFPError(Exception):
    """EFP全体の基底例外クラス。"""

    pass


class APIConnectionError(EFPError):
    """API接続エラー"""

    pass


class DatabaseConnectionError(EFPError):
    """データベース接続エラー"""

    pass


class DataValidationError(EFPError):
    """データ検証エラー"""

    pass


class ConfigurationError(EFPError):
    """設定エラー"""

    pass