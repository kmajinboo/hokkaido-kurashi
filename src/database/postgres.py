"""PostgreSQL storage helpers for EFP data."""

from datetime import date as parse_date

from sqlalchemy import Column, Date, Float, MetaData, Table, create_engine
from sqlalchemy.dialects.postgresql import insert

from src.core.config import DB_CONFIG
from src.core.exceptions import ConfigurationError, DatabaseConnectionError


def _create_engine():
    required = ("host", "port", "database", "user", "password")
    missing = [key for key in required if not DB_CONFIG.get(key)]
    if missing:
        names = ", ".join(f"DB_{key.upper()}" for key in missing)
        raise ConfigurationError(f"Missing database settings: {names}")

    try:
        return create_engine(
            "postgresql+psycopg2://",
            connect_args={
                "host": DB_CONFIG["host"],
                "port": DB_CONFIG["port"],
                "dbname": DB_CONFIG["database"],
                "user": DB_CONFIG["user"],
                "password": DB_CONFIG["password"],
                "connect_timeout": 10,
                # Japanese series names need UTF-8 regardless of the PC's locale
                "client_encoding": "utf8",
            },
            pool_pre_ping=True,
        )
    except Exception as exc:
        raise DatabaseConnectionError("Could not configure PostgreSQL connection") from exc


def save_series(data, table_name: str = "fred_fedfunds") -> int:
    """Upsert date/value observations into a PostgreSQL table."""
    if not table_name.isidentifier():
        raise ValueError("table_name must be a valid SQL identifier")

    metadata = MetaData()
    table = Table(
        table_name,
        metadata,
        Column("date", Date, primary_key=True),
        Column("value", Float, nullable=True),
    )
    rows = [
        {
            "date": parse_date.fromisoformat(str(observation_date)),
            "value": None if value != value else float(value),
        }
        for observation_date, value in zip(data["date"], data["value"])
    ]

    engine = _create_engine()
    try:
        with engine.begin() as connection:
            metadata.create_all(connection)
            if rows:
                statement = insert(table).values(rows)
                statement = statement.on_conflict_do_update(
                    index_elements=[table.c.date],
                    set_={"value": statement.excluded.value},
                )
                connection.execute(statement)
        return len(rows)
    except Exception as exc:
        raise DatabaseConnectionError("Could not save series to PostgreSQL") from exc
    finally:
        engine.dispose()
