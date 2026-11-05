"""Repository interface: where observations and the catalog are stored.

Two implementations:

* ``PostgresRepository`` (standard): tables ``observations`` and ``series``.
* ``CsvRepository``: plain CSV files, for checks without a database.

Both upsert: saving the same series and month again replaces the value.
"""

from abc import ABC, abstractmethod
from datetime import date as parse_date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, Date, DateTime, Float, MetaData, String, Table, select
from sqlalchemy.dialects.postgresql import insert

from src.catalog import SeriesSpec, catalog_frame
from src.core.exceptions import DatabaseConnectionError, DataValidationError

OBSERVATION_COLUMNS = ["series_id", "date", "value", "source", "retrieved_at"]
_CHUNK = 1000


def _validate_observations(observations: pd.DataFrame) -> None:
    missing = set(OBSERVATION_COLUMNS) - set(observations.columns)
    if missing:
        raise DataValidationError(f"Observations are missing columns: {', '.join(sorted(missing))}")
    if observations.duplicated(subset=["series_id", "date"]).any():
        raise DataValidationError("Observations contain the same series and month twice")


class Repository(ABC):
    """Stores the common long-format observations and the series catalog."""

    @abstractmethod
    def save_observations(self, observations: pd.DataFrame) -> int:
        """Upsert rows (series_id, date, value, source, retrieved_at); return the row count."""

    @abstractmethod
    def save_catalog(self, specs: list[SeriesSpec]) -> int:
        """Upsert the series descriptions; return the row count."""

    @abstractmethod
    def load_observations(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        """Return stored observations (all, or only the given series)."""


class PostgresRepository(Repository):
    """Observations in table ``observations``, catalog in table ``series``."""

    def __init__(self, engine_factory=None) -> None:
        if engine_factory is None:
            from src.database.postgres import _create_engine as engine_factory
        self._engine_factory = engine_factory
        self.metadata = MetaData()
        self.observations = Table(
            "observations",
            self.metadata,
            Column("series_id", String(64), primary_key=True),
            Column("date", Date, primary_key=True),
            Column("value", Float, nullable=True),
            Column("source", String(32), nullable=False),
            Column("retrieved_at", DateTime(timezone=True), nullable=False),
        )
        self.series = Table(
            "series",
            self.metadata,
            Column("series_id", String(64), primary_key=True),
            Column("name", String(200), nullable=False),
            Column("unit", String(64)),
            Column("frequency", String(8)),
            Column("layer", String(32)),
            Column("category", String(32)),
            Column("source", String(32)),
            Column("source_id", String(128)),
        )

    def _run(self, work):
        engine = self._engine_factory()
        try:
            with engine.begin() as connection:
                self.metadata.create_all(connection)
                return work(connection)
        except (DataValidationError, DatabaseConnectionError):
            raise
        except Exception as exc:
            raise DatabaseConnectionError("PostgreSQL operation failed") from exc
        finally:
            engine.dispose()

    def save_observations(self, observations: pd.DataFrame) -> int:
        _validate_observations(observations)
        rows = [
            {
                "series_id": str(r.series_id),
                "date": parse_date.fromisoformat(str(r.date)[:10]),
                "value": None if pd.isna(r.value) else float(r.value),
                "source": str(r.source),
                "retrieved_at": datetime.fromisoformat(str(r.retrieved_at)),
            }
            for r in observations.itertuples(index=False)
        ]

        def work(connection):
            for start in range(0, len(rows), _CHUNK):
                statement = insert(self.observations).values(rows[start : start + _CHUNK])
                statement = statement.on_conflict_do_update(
                    index_elements=[self.observations.c.series_id, self.observations.c.date],
                    set_={
                        "value": statement.excluded.value,
                        "source": statement.excluded.source,
                        "retrieved_at": statement.excluded.retrieved_at,
                    },
                )
                connection.execute(statement)
            return len(rows)

        return self._run(work) if rows else 0

    def save_catalog(self, specs: list[SeriesSpec]) -> int:
        rows = catalog_frame(specs).to_dict(orient="records")

        def work(connection):
            if rows:
                statement = insert(self.series).values(rows)
                statement = statement.on_conflict_do_update(
                    index_elements=[self.series.c.series_id],
                    set_={c: statement.excluded[c] for c in rows[0] if c != "series_id"},
                )
                connection.execute(statement)
            return len(rows)

        return self._run(work)

    def load_observations(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        def work(connection):
            query = select(self.observations).order_by(self.observations.c.series_id, self.observations.c.date)
            if series_ids is not None:
                query = query.where(self.observations.c.series_id.in_(series_ids))
            return connection.execute(query).fetchall()

        result = pd.DataFrame(self._run(work), columns=OBSERVATION_COLUMNS)
        if not result.empty:
            result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
            # Always UTC, whatever time zone the database server uses
            result["retrieved_at"] = result["retrieved_at"].map(
                lambda t: t.astimezone(timezone.utc).isoformat(timespec="seconds")
            )
        return result


class CsvRepository(Repository):
    """``observations.csv`` and ``series.csv`` in one directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.observations_path = self.directory / "observations.csv"
        self.series_path = self.directory / "series.csv"

    def _upsert(self, path: Path, new: pd.DataFrame, keys: list[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_csv(path, dtype={k: str for k in keys}, encoding="utf-8-sig")
            new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=keys, keep="last")
        new.sort_values(keys).to_csv(path, index=False, encoding="utf-8-sig")

    def save_observations(self, observations: pd.DataFrame) -> int:
        _validate_observations(observations)
        self._upsert(self.observations_path, observations[OBSERVATION_COLUMNS], ["series_id", "date"])
        return len(observations)

    def save_catalog(self, specs: list[SeriesSpec]) -> int:
        frame = catalog_frame(specs)
        self._upsert(self.series_path, frame, ["series_id"])
        return len(frame)

    def load_observations(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        if not self.observations_path.exists():
            return pd.DataFrame(columns=OBSERVATION_COLUMNS)
        result = pd.read_csv(self.observations_path, dtype={"series_id": str, "date": str}, encoding="utf-8-sig")
        if series_ids is not None:
            result = result[result["series_id"].isin(series_ids)]
        return result.reset_index(drop=True)
