# -*- coding: utf-8 -*-
"""Данные уровней воды: CSV, Excel, SQLite."""

from __future__ import annotations

import os
import psycopg2
import os

def get_db_connection():
    db_url = os.environ.get('DATASET_DATABASE_URL', os.environ.get('DATABASE_URL', 'postgresql://portal_user:admin135@postgres:5432/portal_db'))
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS dataset_schema"
        cur.execute("SET search_path TO dataset_schema"
    return conn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

_APP_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _APP_ROOT / "data"
_DEFAULT_XLSX = _APP_ROOT / "Реки" / "данные январь" / "Уровни_воды.xlsx"
_DEFAULT_CSV = _DATA_DIR / "Уровни_воды.csv"
_DEFAULT_DB = _DATA_DIR / "water_levels.db"

COL_RIVER = "Река"
COL_POST = "Гидрометеорологический пост"
COL_DATE = "Дата (дд.мм..гг)"
COL_LEVEL = "Уровень_воды"

USECOLS = [COL_RIVER, COL_POST, COL_DATE, COL_LEVEL]
FORECAST_DATE = "Дата"
FORECAST_LEVEL = "Уровень_воды"

SQLITE_TABLE = "water_levels"


@dataclass(frozen=True)
class WaterDataSource:
    """Описание источника для UI."""

    source_id: str
    label: str
    path: Path
    kind: str  # csv | xlsx | sqlite


def resolve_january_dir() -> Path:
    raw = os.environ.get("JANUARY_DATA_DIR", "").strip()
    return Path(raw) if raw else _APP_ROOT / "Реки" / "данные январь"


def resolve_default_csv() -> Path:
    raw = os.environ.get("WATER_LEVELS_CSV", "").strip()
    return Path(raw) if raw else _DEFAULT_CSV


def resolve_default_xlsx() -> Path:
    raw = os.environ.get("WATER_LEVELS_XLSX", "").strip()
    if raw:
        return Path(raw)
    candidate = resolve_january_dir() / "Уровни_воды.xlsx"
    return candidate if candidate.is_file() else _DEFAULT_XLSX


def resolve_default_db() -> Path:
    raw = os.environ.get("WATER_LEVELS_DB", "").strip()
    return Path(raw) if raw else _DEFAULT_DB


def resolve_water_levels_path() -> Path:
    """Обратная совместимость: предпочтение CSV, затем Excel."""
    csv_p = resolve_default_csv()
    if csv_p.is_file():
        return csv_p
    return resolve_default_xlsx()


def is_water_levels_available() -> bool:
    return bool(discover_sources())


def discover_sources() -> list[WaterDataSource]:
    """Все доступные источники для выбора в UI."""
    found: list[WaterDataSource] = []
    seen: set[str] = set()

    def add(kind: str, path: Path, label: str, sid: str) -> None:
        if path.name.startswith("~$"):
            return
        key = str(path.resolve())
        if key in seen or not path.is_file():
            return
        seen.add(key)
        found.append(WaterDataSource(source_id=sid, label=label, path=path, kind=kind))

    csv_default = resolve_default_csv()
    add("csv", csv_default, f"CSV: data/{csv_default.name}", "csv_default")

    if _DATA_DIR.is_dir():
        for p in sorted(_DATA_DIR.glob("*.csv"), key=lambda x: x.name.lower()):
            if "уровн" in p.name.lower() or p.name == csv_default.name:
                add("csv", p, f"CSV: data/{p.name}", f"csv_{p.stem}")

    xlsx_default = resolve_default_xlsx()
    add("xlsx", xlsx_default, f"Excel: {xlsx_default.name}", "xlsx_default")

    jan = resolve_january_dir()
    if jan.is_dir():
        for p in sorted(jan.glob("*.xlsx"), key=lambda x: x.name.lower()):
            if "уровн" in p.name.lower():
                add("xlsx", p, f"Excel: {p.relative_to(_APP_ROOT)}", f"xlsx_{p.stem}")

    db = resolve_default_db()
    if db.is_file() and _sqlite_has_table(db):
        add("sqlite", db, f"SQLite: data/{db.name}", "sqlite_default")

    hydro_db = os.environ.get("HYDRO_METEO_DB", "").strip()
    if hydro_db:
        hp = Path(hydro_db)
        if hp.is_file() and _sqlite_has_water_levels(hp):
            add("sqlite", hp, f"SQLite: {hp.name} (уровни)", f"sqlite_hydro_{hp.stem}")

    return found


def _sqlite_has_table(db_path: Path, table: str = SQLITE_TABLE) -> bool:
    try:
        with get_db_connection()) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=%s",
                (table,,
            ).fetchone()
            return row is not None
    except psycopg2.Error:
        return False


def _sqlite_has_water_levels(db_path: Path) -> bool:
    return _sqlite_has_table(db_path, SQLITE_TABLE)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Привести к стандартным колонкам."""
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == COL_RIVER.lower() or c == COL_RIVER:
            rename_map[c] = COL_RIVER
        elif "гидрометеоролог" in cl or c == COL_POST:
            rename_map[c] = COL_POST
        elif "дата" in cl and "уров" not in cl:
            rename_map[c] = COL_DATE
        elif "уровень" in cl or c == COL_LEVEL:
            rename_map[c] = COL_LEVEL

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in USECOLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле нет колонок: {', '.join(missing)}. "
            f"Найдены: {', '.join(str(c) for c in df.columns[:15])}"
        )

    out = df[USECOLS].copy()
    out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce", dayfirst=True)
    out[COL_LEVEL] = pd.to_numeric(out[COL_LEVEL], errors="coerce")
    return out.dropna(subset=[COL_RIVER, COL_DATE, COL_LEVEL])


def load_water_levels_table(
    path: Optional[Path] = None,
    *,
    kind: Optional[str] = None,
) -> pd.DataFrame:
    """Загрузка из CSV, Excel или SQLite."""
    p = Path(path) if path else resolve_water_levels_path()
    if not p.is_file():
        raise FileNotFoundError(f"Файл не найден: {p}")

    ext = p.suffix.lower()
    resolved_kind = kind or ("sqlite" if ext == ".db" else ext.lstrip("."))

    if resolved_kind == "sqlite":
        return _load_sqlite(p)
    if resolved_kind in ("csv", "txt"):
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        return _normalize_frame(df)
    if resolved_kind in ("xlsx", "xls", "xlsm"):
        try:
            df = pd.read_excel(p, usecols=USECOLS)
        except ValueError:
            df = pd.read_excel(p)
        return _normalize_frame(df)

    raise ValueError(f"Неподдерживаемый формат: {p.suffix}")


def _load_sqlite(db_path: Path) -> pd.DataFrame:
    sql = f"""
        SELECT
            river AS [{COL_RIVER}],
            post AS [{COL_POST}],
            observation_date AS [{COL_DATE}],
            level AS [{COL_LEVEL}]
        FROM {SQLITE_TABLE}
    """
    with get_db_connection()) as conn:
        df = pd.read_sql_query(sql, conn)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    df[COL_LEVEL] = pd.to_numeric(df[COL_LEVEL], errors="coerce")
    return df.dropna(subset=[COL_RIVER, COL_DATE, COL_LEVEL])


def build_sqlite_from_csv(
    csv_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    *,
    replace: bool = True,
) -> Path:
    """Создать/обновить data/water_levels.db из CSV."""
    csv_p = Path(csv_path) if csv_path else resolve_default_csv()
    db_p = Path(db_path) if db_path else resolve_default_db()
    if not csv_p.is_file():
        raise FileNotFoundError(f"CSV не найден: {csv_p}")

    df = load_water_levels_table(csv_p, kind="csv")
    db_p.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection()) as conn:
        if replace:
            conn.execute(f"DROP TABLE IF EXISTS {SQLITE_TABLE}"
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SQLITE_TABLE} (
                id SERIAL PRIMARY KEY,
                river TEXT NOT NULL,
                post TEXT,
                observation_date TEXT NOT NULL,
                level REAL,
                UNIQUE(river, post, observation_date
            )
            """
        )
        rows = [
            (
                str(r[COL_RIVER]),
                str(r[COL_POST]) if pd.notna(r[COL_POST]) else "",
                r[COL_DATE].strftime("%Y-%m-%d") if pd.notna(r[COL_DATE]) else "",
                float(r[COL_LEVEL]) if pd.notna(r[COL_LEVEL]) else None,
            )
            for _, r in df.iterrows()
        ]
        conn.executemany(
            f"""
            INSERT INTO {SQLITE_TABLE}
            (river, post, observation_date, level
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return db_p


def list_rivers(df: pd.DataFrame) -> list[str]:
    rivers = sorted(df[COL_RIVER].astype(str).unique(), key=str.casefold)
    return rivers


def list_posts_for_river(df: pd.DataFrame, river: str) -> list[str]:
    sub = df[df[COL_RIVER] == river]
    posts = sorted(sub[COL_POST].astype(str).unique(), key=str.casefold)
    return posts


def build_river_series(
    df: pd.DataFrame,
    river: str,
    *,
    post: Optional[str] = None,
    aggregate_posts: bool = True,
) -> pd.DataFrame:
    """Ряд для прогноза: Дата + Уровень_воды."""
    sub = df[df[COL_RIVER] == river].copy()
    if post:
        sub = sub[sub[COL_POST] == post]
    if sub.empty:
        return pd.DataFrame(columns=[FORECAST_DATE, FORECAST_LEVEL])

    out = (
        sub.groupby(COL_DATE, as_index=False)[COL_LEVEL]
        .mean()
        .rename(columns={COL_DATE: FORECAST_DATE, COL_LEVEL: FORECAST_LEVEL})
    )
    out = out.sort_values(FORECAST_DATE).reset_index(drop=True)
    out[FORECAST_DATE] = pd.to_datetime(out[FORECAST_DATE], errors="coerce")
    out[FORECAST_LEVEL] = pd.to_numeric(out[FORECAST_LEVEL], errors="coerce")
    return out.dropna(subset=[FORECAST_DATE, FORECAST_LEVEL])
