"""
EFP Configuration Module

プロジェクト全体の設定を管理します。
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# -------------------------------
# プロジェクトルート
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# .env を読み込む
load_dotenv(BASE_DIR / ".env")

# -------------------------------
# Database
# -------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# -------------------------------
# API Keys
# -------------------------------
API_KEYS = {
    "fred": os.getenv("FRED_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
}
AI_PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()
AI_MODEL = os.getenv("AI_MODEL")

# -------------------------------
# Directories
# -------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature"
PREDICTION_DIR = DATA_DIR / "prediction"

LOG_DIR = BASE_DIR / "logs"

OUTPUT_DIR = BASE_DIR / "output"
POWERBI_DIR = BASE_DIR / "powerbi"

MODEL_DIR = BASE_DIR / "models"

# -------------------------------
# Default Settings
# -------------------------------
DEFAULT_COUNTRY = "US"

DEFAULT_LANGUAGE = "ja"

DEFAULT_TIMEZONE = "Asia/Tokyo"