# -*- coding: utf-8 -*-
"""Вкладка: прогноз уровней воды (выбор файла / базы данных)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ..water_levels import (
    FORECAST_DATE,
    FORECAST_LEVEL,
    WaterDataSource,
    build_river_series,
    build_sqlite_from_csv,
    discover_sources,
    list_posts_for_river,
    list_rivers,
    load_water_levels_table,
    resolve_default_csv,
    resolve_default_db,
)

_chart_id = 0


def _plot_key() -> str:
    global _chart_id
    _chart_id += 1
    return f"water_chart_{_chart_id}"


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner="Загрузка данных уровней воды…")
def _cached_water_table(source_key: str, path_str: str, kind: str, mtime: float) -> pd.DataFrame:
    del source_key, mtime
    return load_water_levels_table(Path(path_str), kind=kind)


def _render_source_picker() -> tuple[pd.DataFrame | None, str]:
    """Выбор источника. Возвращает (dataframe, подпись источника)."""
    sources = discover_sources()

    st.subheader("📂 Источник данных")
    mode = st.radio(
        "Откуда брать уровни воды",
        [
            "Из списка (CSV / Excel / база)",
            "Загрузить свой файл",
            "Собрать SQLite из CSV (data/)",
        ],
        horizontal=False,
        key="water_source_mode",
    )

    if mode == "Собрать SQLite из CSV (data/)":
        csv_p = resolve_default_csv()
        db_p = resolve_default_db()
        st.caption(
            f"Создаёт `{db_p.name}` в папке `data/` из CSV. "
            "Удобно для быстрого доступа без повторного чтения Excel."
        )
        if not csv_p.is_file():
            st.error(f"Сначала нужен CSV: `{csv_p}`")
            return None, ""
        if st.button("Собрать базу данных", type="primary", key="water_build_db"):
            with st.spinner("Импорт в SQLite…"):
                try:
                    out = build_sqlite_from_csv(csv_p, db_p)
                    st.success(f"Готово: `{out}` ({out.stat().st_size // 1024} КБ)")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(str(e))
        if db_p.is_file():
            st.info(f"База уже есть: `{db_p}` — выберите режим «Из списка».")
        return None, ""

    if mode == "Загрузить свой файл":
        up = st.file_uploader(
            "CSV или Excel с колонками: Река, Гидрометеорологический пост, "
            "Дата (дд.мм..гг), Уровень_воды",
            type=["csv", "xlsx", "xls"],
            key="water_upload",
        )
        if up is None:
            st.info("Выберите файл для расчёта прогноза.")
            return None, ""
        suffix = Path(up.name).suffix.lower()
        kind = "csv" if suffix == ".csv" else "xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.getbuffer())
            tmp_path = Path(tmp.name)
        try:
            df = load_water_levels_table(tmp_path, kind=kind)
            return df, f"Загружено: {up.name}"
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
            return None, ""
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Из списка
    if not sources:
        st.error(
            "Нет готовых источников. Положите `Уровни_воды.csv` в `dataset-app/data/` "
            "или `Уровни_воды.xlsx` в `Реки/данные январь/`."
        )
        return None, ""

    labels = [s.label for s in sources]
    default_idx = 0
    for i, s in enumerate(sources):
        if s.kind == "csv":
            default_idx = i
            break

    choice = st.selectbox(
        "Файл или база",
        range(len(sources)),
        index=default_idx,
        format_func=lambda i: labels[i],
        key="water_source_pick",
    )
    src: WaterDataSource = sources[choice]
    st.caption(f"Путь: `{src.path}` · тип: {src.kind.upper()}")

    try:
        df = _cached_water_table(
            src.source_id,
            str(src.path),
            src.kind,
            _file_mtime(src.path),
        )
        return df, src.label
    except Exception as e:
        st.error(f"Ошибка чтения: {e}")
        return None, ""


def render_water_forecast_page() -> None:
    st.markdown(
        "<h2 style='color: #F8FAFC; margin-bottom: 8px;'>🌊 Прогноз уровней воды</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Отдельный режим: выберите CSV, Excel или SQLite, затем реку и гидропост. "
        "Не смешивается с остальными вкладками портала."
    )

    raw, source_label = _render_source_picker()
    if raw is None or raw.empty:
        return

    st.success(f"Источник: **{source_label}** · записей: **{len(raw):,}**".replace(",", " "))

    rivers = list_rivers(raw)
    if not rivers:
        st.warning("В выбранном источнике нет рек.")
        return

    st.markdown("---")
    st.subheader("🗺️ Река и гидропост")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        river = st.selectbox("Река", rivers, key="water_river")
    posts = list_posts_for_river(raw, river)
    with c2:
        post_options = ["— Среднее по всем постам реки —"] + posts
        post_choice = st.selectbox("Гидропост", post_options, key="water_post")
    with c3:
        st.metric("Постов", len(posts))

    post = None if post_choice.startswith("—") else post_choice
    series = build_river_series(raw, river, post=post, aggregate_posts=post is None)

    if series.empty or len(series) < 30:
        st.warning(
            f"Мало данных для «{river}»"
            + (f" / {post}" if post else "")
            + f" ({len(series)} точек). Нужно ≥30 для обучения."
        )
        if not series.empty:
            st.dataframe(series.tail(20), use_container_width=True)
        return

    label = f"{river}" + (f" / {post}" if post else " (среднее по постам)")
    st.session_state["water_series_label"] = label
    st.session_state["water_levels_df"] = series

    dmin, dmax = series[FORECAST_DATE].min(), series[FORECAST_DATE].max()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Точек", len(series))
    m2.metric("С", dmin.strftime("%d.%m.%Y") if pd.notna(dmin) else "—")
    m3.metric("По", dmax.strftime("%d.%m.%Y") if pd.notna(dmax) else "—")
    m4.metric(
        "Уровень (см)",
        f"{series[FORECAST_LEVEL].min():.0f} – {series[FORECAST_LEVEL].max():.0f}",
    )

    fig = px.line(
        series,
        x=FORECAST_DATE,
        y=FORECAST_LEVEL,
        title=f"Уровень воды: {label}",
        labels={FORECAST_DATE: "Дата", FORECAST_LEVEL: "Уровень, см"},
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True, key=_plot_key())

    st.markdown("---")
    st.subheader("🔮 Прогнозная модель")

    if "predictor" not in st.session_state:
        st.warning("Предиктор не инициализирован. Обновите страницу.")
        return

    predictor = st.session_state.predictor
    date_col = FORECAST_DATE
    target_col = FORECAST_LEVEL

    c_set1, c_set2 = st.columns(2)
    with c_set1:
        model_type = st.selectbox(
            "Модель",
            ["xgboost", "linear"],
            format_func=lambda x: "XGBoost" if x == "xgboost" else "Линейная",
            key="water_model_type",
        )
    with c_set2:
        forecast_type = st.selectbox(
            "Тип прогноза",
            ["На следующий год", "На конкретную дату"],
            key="water_forecast_type",
        )

    specific_date = None
    if forecast_type == "На конкретную дату":
        specific_date = st.date_input(
            "Дата прогноза",
            value=datetime.now().date() + timedelta(days=30),
            key="water_specific_date",
        )

    if st.button("🚀 Обучить и построить прогноз", type="primary", key="water_train_btn"):
        with st.spinner("Обучение модели…"):
            try:
                score = predictor.train_model(series, target_col, date_col, model_type)
                st.success(f"Модель обучена (R² = {score:.3f})")

                if forecast_type == "На следующий год":
                    fcast = None
                    if hasattr(predictor, "create_visual_forecast"):
                        fcast = predictor.create_visual_forecast(
                            series, target_col, date_col
                        )
                    elif hasattr(predictor, "create_forecast_plot"):
                        fcast = predictor.create_forecast_plot(
                            series, target_col, date_col
                        )
                    if fcast:
                        st.plotly_chart(fcast, use_container_width=True, key=_plot_key())
                    if hasattr(predictor, "create_detailed_forecast_report"):
                        report = predictor.create_detailed_forecast_report(
                            series, target_col, date_col
                        )
                        if report and report.get("summary"):
                            st.info(report["summary"])
                else:
                    result = None
                    if hasattr(predictor, "predict_for_specific_date_fixed"):
                        result = predictor.predict_for_specific_date_fixed(
                            series,
                            target_col,
                            date_col,
                            specific_date.strftime("%d.%m.%Y"),
                        )
                    elif hasattr(predictor, "predict_for_specific_date"):
                        result = predictor.predict_for_specific_date(
                            series,
                            target_col,
                            date_col,
                            specific_date.strftime("%d.%m.%Y"),
                        )
                    if result:
                        st.metric("Прогноз уровня (см)", f"{result.get('value', 0):.1f}")
                        if result.get("date"):
                            st.caption(f"Опора на данные: {result.get('date')}")

            except Exception as e:
                st.error(f"Ошибка: {e}")
