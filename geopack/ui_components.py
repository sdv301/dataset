# ui_components.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import folium
from .analyzer import GeoDataAnalyzer
from .visualization import Visualizer
from .predictor import RiverDataPredictor
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import folium

def auto_load_data():
    """Автоматическая загрузка данных паводков при старте"""
    if 'auto_loaded' not in st.session_state:
        st.session_state.auto_loaded = True
        target_file = "Данные_Среднеколымск Общее_очищенный.xlsx"
        
        # Проверяем, есть ли такой файл в папке
        if 'analyzer' in st.session_state:
            files = st.session_state.analyzer.get_data_files()
            file_names = [f.name for f in files]
            
            if target_file in file_names:
                # Загружаем файл тихо
                file_path = Path(st.session_state.analyzer.data_folder) / target_file
                df = st.session_state.analyzer.load_file(file_path)
                
                if df is not None and not df.empty:
                    # Успешная загрузка — обрабатываем (без st.error/st.success чтобы не засорять UI)
                    st.session_state['current_df'] = df
                    st.session_state['current_file'] = target_file
                    st.session_state['current_file_path'] = str(file_path)
                    
                    df_processed = st.session_state.analyzer.detect_and_convert_dtypes(df)
                    if df_processed is not None and not df_processed.empty:
                        df = df_processed
                    st.session_state['current_df'] = df
                    
                    # Проверяем наличие координат
                    has_lat = any(col.lower() in ['широта', 'lat', 'latitude'] for col in df.columns)
                    has_lon = any(col.lower() in ['долгота', 'lon', 'longitude', 'lng'] for col in df.columns)
                    
                    if has_lat and has_lon:
                        gdf = st.session_state.analyzer.create_geodataframe(df)
                        if gdf is not None and not gdf.empty:
                            st.session_state['current_gdf'] = gdf
                            st.session_state['has_coords'] = True
                            st.session_state['lat_col'] = next(col for col in df.columns if col.lower() in ['широта', 'lat', 'latitude'])
                            st.session_state['lon_col'] = next(col for col in df.columns if col.lower() in ['долгота', 'lon', 'longitude', 'lng'])
                    else:
                        st.session_state['has_coords'] = False

def setup_app():
    """Настройка приложения"""
    st.set_page_config(
        page_title="Анализатор данных с геоаналитикой",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Инициализация компонентов в session_state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GeoDataAnalyzer()
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
    if 'predictor' not in st.session_state:
        st.session_state.predictor = RiverDataPredictor()
    
    # Автоматическая предзагрузка паводков
    auto_load_data()
    
    # CSS стили с улучшенными подсказками
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin: 10px 0;
        }
        .plot-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Улучшенные подсказки Plotly */
        .hoverlayer {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 8px !important;
            border: 1px solid #ccc !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        .hovertext {
            font-family: Arial, sans-serif !important;
            font-size: 12px !important;
            color: #333 !important;
            background-color: white !important;
            padding: 8px !important;
            border-radius: 4px !important;
            border: 1px solid #ddd !important;
        }
        .ytooltip {
            background-color: white !important;
            color: #333 !important;
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
            padding: 8px !important;
        }
        
        /* Настройка для темной темы */
        .js-plotly-plot .plotly .modebar {
            background-color: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title(" Анализатор данных с расширенной геоаналитикой")
    st.markdown("---")

def render_sidebar():
    """Рендер сайдбара"""
    # Убедимся, что анализатор инициализирован
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GeoDataAnalyzer()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = RiverDataPredictor()
    
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Загрузка файлов из папки
        st.subheader("📁 Загрузка данных")
        files = st.session_state.analyzer.get_data_files()
        
        if files:
            selected_file = st.selectbox(
                "Выберите файл из папки 'data':",
                [f.name for f in files],
                index=0
            )
            
            if st.button("📥 Загрузить выбранный файл", type="primary"):
                load_file_from_folder(selected_file)
        
        # Загрузка файла через интерфейс
        st.subheader("Или загрузите файл:")
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['csv', 'xlsx', 'xls', 'parquet', 'json', 'geojson', 'shp'],
            help="Поддерживаются CSV, Excel, Parquet, JSON, GeoJSON, Shapefile"
        )
        
        if uploaded_file is not None:
            load_uploaded_file(uploaded_file)
        
        # Настройки темы визуализации
        st.markdown("---")
        st.subheader("🎨 Настройки визуализации")
        
        # Настройки темы
        theme_options = ['plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']
        selected_theme = st.selectbox(
            "Тема оформления",
            theme_options,
            index=0
        )
        
        if st.button("🎨 Применить тему"):
            st.session_state.visualizer.theme = selected_theme
            st.success(f"Тема '{selected_theme}' применена")
        
        # Настройки подсказок
        st.markdown("---")
        st.subheader("💡 Настройки подсказок")
        
        hover_bg_color = st.color_picker("Цвет фона подсказки", "#FFFFFF")
        hover_text_color = st.color_picker("Цвет текста подсказки", "#000000")
        hover_font_size = st.slider("Размер шрифта подсказки", 8, 16, 12)
        
        if st.button("💡 Применить настройки подсказок"):
            # Обновляем настройки визуализатора
            st.session_state.visualizer.default_config['hoverlabel'] = {
                'bgcolor': hover_bg_color,
                'font_size': hover_font_size,
                'font_color': hover_text_color
            }
            st.success("Настройки подсказок применены")
        
        # Гео-инструменты
        st.markdown("---")
        st.subheader("🗺️ Геоинструменты")
        
        if st.session_state.get('has_coords', False):
            st.success("✅ Геоданные доступны")
            
            crs_options = {
                "WGS84 (широта/долгота)": "EPSG:4326",
                "Меркатор (веб-карты)": "EPSG:3857",
                "Пулково 1942": "EPSG:4284",
                "СК-42": "EPSG:4282"
            }
            
            selected_crs = st.selectbox(
                "Система координат",
                list(crs_options.keys())
            )
            
            if st.button("🔄 Применить систему координат"):
                if 'current_gdf' in st.session_state:
                    st.session_state['current_gdf'] = st.session_state['current_gdf'].to_crs(crs_options[selected_crs])
                    st.success(f"Применена система: {selected_crs}")
        
        st.markdown("---")
        st.caption("🌍 Поддерживаются: CSV, Excel, GeoJSON, Shapefiles")
        
        # Информация о загруженных данных
        if 'current_file' in st.session_state:
            st.markdown("---")
            has_coords = st.session_state.get('has_coords', False)
            data_type = "🌍 Геоданные" if has_coords else "📊 Обычные данные"
            
            st.info(f"""
            **Загружен файл:** {st.session_state['current_file']}
            
            **Тип данных:** {data_type}
            
            **Координаты:** {'✅ Доступны' if has_coords else '❌ Не обнаружены'}
            
            **Время загрузки:** {datetime.now().strftime('%H:%M:%S')}
            """)

def render_main_content():
    """Рендер основного контента"""
    # Проверяем наличие любых данных
    has_any_data = (
        'current_gdf' in st.session_state or 
        'current_df' in st.session_state or 
        'special_coords_df' in st.session_state
    )
    
    if not has_any_data:
        render_start_page()
    else:
        render_data_tabs()

def render_start_page():
    """Стартовая страница"""
    st.info("👈 Загрузите данные через сайдбар или положите файлы в папку 'data/'")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📁 Поддерживаемые форматы")
        st.markdown("""
        - **CSV/Excel** с координатами
        - **GeoJSON** - стандарт геоданных
        - **Shapefile** - ESRI формат
        - **Parquet** - быстрый бинарный
        - **JSON** - структурированный
        """)
    
    with col2:
        st.markdown("### 📊 Анализ данных")
        st.markdown("""
        - Статистический анализ
        - Пространственный анализ
        - Кластеризация
        - Визуализация
        """)
    
    with col3:
        st.markdown("### 🗺️ Геоаналитика")
        st.markdown("""
        - Тепловые карты
        - Буферные зоны
        - Диаграмма Вороного
        - Пространственные запросы
        """)
    
    # Пример геоданных
    st.markdown("### Пример геоданных (GeoJSON)")
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Москва", "population": 12615},
                "geometry": {"type": "Point", "coordinates": [37.6173, 55.7558]}
            },
            {
                "type": "Feature",
                "properties": {"name": "СПб", "population": 5384},
                "geometry": {"type": "Point", "coordinates": [30.3351, 59.9343]}
            }
        ]
    }
    st.json(sample_geojson, expanded=False)

def render_data_tabs():
    """Рендер вкладок с данными"""
    st.session_state.get('current_file', 'Файл')
    has_coords = st.session_state.get('has_coords', False)
    has_special_coords = st.session_state.get('has_special_coords', False)
    
    # Определяем тип данных
    if has_coords and 'current_gdf' in st.session_state:
        pass
    elif has_special_coords and 'special_coords_df' in st.session_state:
        pass
    else:
        pass
    
    # Создаем вкладки
    tabs = ["📋 Обзор данных", "🛠️ Корректировка типов", "📊 Визуализация"]
    
    # Проверяем, есть ли временные данные для предсказаний
    has_time_data = False
    df = None
    
    if 'current_df' in st.session_state:
        df = st.session_state['current_df']
    elif 'special_coords_df' in st.session_state:
        df = st.session_state['special_coords_df']
    
    if df is not None and not df.empty:
        # Проверяем наличие дат
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
        
        has_time_data = len(date_cols) > 0
    
    # Добавляем вкладку предсказаний если есть временные данные
    if has_time_data:
        tabs.append("🔮 Предсказания")
    
    if has_time_data:
        tabs.append("🌊 Пороговый анализ")
    
    # Добавляем карту и русские карты только если есть геоданные
    if has_coords:
        tabs.extend(["🗺️ Карта", "🌍 Геоаналитика", "📈 Дашборд"])
    elif has_special_coords:
        tabs.extend(["📍 Специальные координаты", "📈 Дашборд"])
    else:
        tabs.extend(["📈 Дашборд"])

    tab_objects = st.tabs(tabs)
    
    # Вкладка 1: Обзор данных
    with tab_objects[0]:
        show_data_overview()

    # Вкладка 2: Корректировка типов
    with tab_objects[1]:
        show_data_type_correction()

    # Вкладка 3: Визуализация
    with tab_objects[2]:
        show_visualization()

    # Определяем текущий индекс
    current_tab_index = 3  # После трех базовых вкладок

    # Вкладка: Предсказания (если есть временные данные)
    if has_time_data:
        if current_tab_index < len(tab_objects):
            with tab_objects[current_tab_index]:
                show_predictions()
            current_tab_index += 1

    # Вкладка: Пороговый анализ (если есть временные данные)
    if has_time_data:
        if current_tab_index < len(tab_objects):
            with tab_objects[current_tab_index]:
                show_threshold_analysis()
            current_tab_index += 1

    # Вкладка: Специальные координаты
    if has_special_coords:
        if current_tab_index < len(tab_objects):
            with tab_objects[current_tab_index]:
                show_special_coordinates()
            current_tab_index += 1

    # Вкладка: Карта (только для геоданных)
    if has_coords:
        map_tab_index = tabs.index("🗺️ Карта") if "🗺️ Карта" in tabs else current_tab_index
        if map_tab_index < len(tab_objects):
            with tab_objects[map_tab_index]:
                show_map_view()
            current_tab_index += 1

    # Вкладка: Геоаналитика (только для геоданных)
    if has_coords:
        geo_tab_index = tabs.index("🌍 Геоаналитика") if "🌍 Геоаналитика" in tabs else current_tab_index
        if geo_tab_index < len(tab_objects):
            with tab_objects[geo_tab_index]:
                show_geo_analysis()
            current_tab_index += 1

    # Вкладка: Дашборд (последняя всегда)
    dashboard_tab_index = tabs.index("📈 Дашборд") if "📈 Дашборд" in tabs else current_tab_index
    if dashboard_tab_index < len(tab_objects):
        with tab_objects[dashboard_tab_index]:
            show_dashboard()

def load_file_from_folder(selected_file):
    """Загрузить файл из папки"""
    file_path = Path(st.session_state.analyzer.data_folder) / selected_file
    df = st.session_state.analyzer.load_file(file_path)
    
    if df is not None and not df.empty:
        process_loaded_data(df, selected_file, file_path)
    else:
        st.error(f"Не удалось загрузить файл {selected_file}")

def load_uploaded_file(uploaded_file):
    """Загрузить загруженный файл"""
    try:
        temp_path = Path("temp") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        df = st.session_state.analyzer.load_file(temp_path)
        
        if df is not None and not df.empty:
            process_loaded_data(df, uploaded_file.name, temp_path)
            temp_path.unlink()
        else:
            st.error(f"Не удалось загрузить файл {uploaded_file.name}")
            
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")

def process_loaded_data(df, filename, file_path=None):
    """Обработать загруженные данные"""
    if df is None or df.empty:
        st.error("Файл пустой или не содержит данных")
        st.session_state['current_df'] = None
        st.session_state['has_coords'] = False
        st.session_state['current_file'] = filename
        return
    
    # Автоматически определяем и преобразуем типы данных
    with st.spinner("🔍 Анализирую типы данных..."):
        df_processed = st.session_state.analyzer.detect_and_convert_dtypes(df)
    
    if df_processed is None or df_processed.empty:
        df_processed = df  # Возвращаем оригинальный df если преобразование не удалось
    
    # Сохраняем информацию о файле в session_state для повторного использования
    if file_path:
        st.session_state['current_file_path'] = str(file_path)
    
    # Если это Excel файл, получаем список листов
    if filename.endswith(('.xlsx', '.xls')) and file_path:
        try:
            sheet_names = st.session_state.analyzer.get_excel_sheets(file_path)
            
            if sheet_names and len(sheet_names) > 1:
                st.subheader("📑 Выбор листа Excel")
                
                # Сохраняем список листов в session_state
                st.session_state['excel_sheets_info'] = {
                    'file_path': str(file_path),
                    'sheet_names': sheet_names,
                    'current_sheet': 0  # индекс выбранного листа
                }
                
                # Показываем текущий лист
                current_sheet = st.session_state['excel_sheets_info']['current_sheet']
                st.info(f"Текущий лист: **{sheet_names[current_sheet]}**")
                
                # Используем форму для предотвращения перезагрузки
                with st.form(key="excel_sheet_form"):
                    # Выбор нового листа
                    selected_sheet_index = st.selectbox(
                        "Выберите другой лист:",
                        range(len(sheet_names)),
                        format_func=lambda x: f"{sheet_names[x]} ({x+1}/{len(sheet_names)})",
                        key=f"sheet_select_{filename}",
                        index=current_sheet  # Устанавливаем текущий индекс
                    )
                    
                    # Кнопка для загрузки выбранного листа внутри формы
                    submit_button = st.form_submit_button("🔄 Переключиться на лист")
                    
                    if submit_button:
                        selected_sheet_name = sheet_names[selected_sheet_index]
                        
                        # Загружаем выбранный лист
                        new_df = st.session_state.analyzer.load_excel_sheet(file_path, selected_sheet_name)
                        
                        if new_df is not None and not new_df.empty:
                            # Преобразуем типы данных для нового листа
                            new_df = st.session_state.analyzer.detect_and_convert_dtypes(new_df)
                            
                            # Обновляем данные в session_state
                            st.session_state['current_df'] = new_df
                            st.session_state['excel_sheets_info']['current_sheet'] = selected_sheet_index
                            
                            # Сообщаем об успехе и продолжаем работу
                            st.success(f"✅ Загружен лист: **{selected_sheet_name}**")
                            st.rerun()  # Перезагружаем только для обновления отображения
                        else:
                            st.error(f"Не удалось загрузить лист {selected_sheet_name}")
                
                st.info("ℹ️ Выберите лист и нажмите 'Переключиться на лист' для загрузки")
                
        except Exception as e:
            st.warning(f"Не удалось получить список листов: {e}")
    
    # Показываем информацию о загруженных данных
    st.info(f"📊 Загружено {len(df_processed)} строк, {len(df_processed.columns)} колонок")
    
    # Проверяем, не загрузили ли мы статистику вместо данных
    if len(df_processed.columns) > 0 and 'колонка' in str(df_processed.columns[0]).lower():
        st.error("⚠️ Загружена статистика данных, а не сами данные!")
        st.info("Пожалуйста, загрузите файл с исходными данными, а не с отчетом статистики")
        return
    
    # Проверяем стандартные координаты
    lat_cols, lon_cols = st.session_state.analyzer.detect_coordinates(df_processed)
    
    # Обработка координат
    if lat_cols and lon_cols:
        try:
            st.info(f"📍 Найдены координаты: Широта='{lat_cols[0]}', Долгота='{lon_cols[0]}'")
            
            # Пробуем создать геодатафрейм
            gdf = st.session_state.analyzer.create_geodataframe(df_processed, lat_cols[0], lon_cols[0])
            
            if gdf is not None and not gdf.empty:
                st.session_state['current_gdf'] = gdf
                st.session_state['has_coords'] = True
                st.session_state['lat_col'] = lat_cols[0]
                st.session_state['lon_col'] = lon_cols[0]
                st.session_state['current_df'] = df_processed
                st.success(f"✅ Геоданные созданы! ({len(gdf)} объектов)")
            else:
                st.warning("⚠️ Не удалось создать геодатафрейм из координат")
                st.session_state['current_df'] = df_processed
                st.session_state['has_coords'] = False
                
        except Exception as e:
            st.error(f"Ошибка создания геодатафрейма: {str(e)}")
            st.session_state['current_df'] = df_processed
            st.session_state['has_coords'] = False
    else:
        # Нет координат
        st.info("📍 Координаты не обнаружены автоматически")
        st.session_state['current_df'] = df_processed
        st.session_state['has_coords'] = False
    
    # Всегда сохраняем информацию о файле
    st.session_state['current_file'] = filename
    
    # Финальное сообщение
    if 'excel_sheets_info' in st.session_state:
        sheet_info = st.session_state['excel_sheets_info']
        current_idx = sheet_info.get('current_sheet', 0)
        current_sheet = sheet_info['sheet_names'][current_idx]
        st.success(f"✅ Файл '{filename}' (лист '{current_sheet}') загружен успешно!")
    else:
        st.success(f"✅ Файл '{filename}' загружен успешно!")

def render_excel_sheet_selector():
    """Рендер селектора листов Excel для уже загруженного файла"""
    if 'excel_sheets_info' in st.session_state and 'current_file_path' in st.session_state:
        sheet_info = st.session_state['excel_sheets_info']
        file_path = st.session_state['current_file_path']
        filename = st.session_state.get('current_file', '')
        
        if sheet_info['sheet_names'] and len(sheet_info['sheet_names']) > 1:
            st.markdown("---")
            st.subheader("🔄 Переключение листов Excel")
            
            current_idx = sheet_info.get('current_sheet', 0)
            current_sheet = sheet_info['sheet_names'][current_idx]
            
            st.info(f"Текущий лист: **{current_sheet}**")
            
            # Используем форму
            with st.form(key="switch_excel_sheet_form"):
                # Полный выбор через selectbox
                selected_sheet_index = st.selectbox(
                    "Выберите лист:",
                    range(len(sheet_info['sheet_names'])),
                    format_func=lambda x: f"{sheet_info['sheet_names'][x]} ({x+1}/{len(sheet_info['sheet_names'])})",
                    key=f"full_sheet_select_{filename}",
                    index=current_idx  # Устанавливаем текущий индекс
                )
                
                submit_button = st.form_submit_button("🔄 Переключиться на выбранный лист")
                
                if submit_button:
                    selected_sheet_name = sheet_info['sheet_names'][selected_sheet_index]
                    
                    # Загружаем выбранный лист
                    new_df = st.session_state.analyzer.load_excel_sheet(file_path, selected_sheet_name)
                    
                    if new_df is not None and not new_df.empty:
                        # Преобразуем типы данных
                        new_df = st.session_state.analyzer.detect_and_convert_dtypes(new_df)
                        
                        st.session_state['current_df'] = new_df
                        st.session_state['excel_sheets_info']['current_sheet'] = selected_sheet_index
                        
                        st.success(f"✅ Переключено на лист: **{selected_sheet_name}**")
                        st.rerun()  # Перезагружаем для обновления данных
                    else:
                        st.error(f"Не удалось загрузить лист {selected_sheet_name}")

def show_data_overview():
    """Показать вкладку обзора данных"""
    st.header("📋 Обзор данных")
    
    # Добавляем селектор листов Excel, если файл Excel
    if st.session_state.get('current_file', '').endswith(('.xlsx', '.xls')):
        render_excel_sheet_selector()
    
    # Получаем данные из сессии
    has_coords = st.session_state.get('has_coords', False)
    df = None
    gdf = None
    
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        if not gdf.empty:
            df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    elif 'current_df' in st.session_state:
        df = st.session_state['current_df']
    
    if df is None or df.empty:
        st.warning("Нет данных для отображения")
        return
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Строки", len(df))
    with col2:
        st.metric("Колонки", len(df.columns))
    with col3:
        st.metric("Пропущено", df.isnull().sum().sum())
    with col4:
        if gdf is not None:
            st.metric("Геообъектов", len(gdf))
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Числовых колонок", len(numeric_cols))
    
    # Предпросмотр данных
    st.subheader("Предпросмотр данных")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Информация о колонках
    st.subheader("Информация о колонках")
    col_info = st.session_state.analyzer.get_column_statistics(df)
    st.dataframe(col_info, use_container_width=True)
    
    # Если есть геоданные
    if gdf is not None and not gdf.empty:
        st.subheader("🌍 Геометрическая информация")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Типы геометрий:**")
            st.write(gdf.geometry.type.value_counts())
        
        with col2:
            st.write("**Границы данных:**")
            bounds = gdf.total_bounds
            st.write(f"Широта: {bounds[1]:.4f} - {bounds[3]:.4f}")
            st.write(f"Долгота: {bounds[0]:.4f} - {bounds[2]:.4f}")
    
    # Статистика
    st.subheader("Базовая статистика")
    numeric_stats = st.session_state.analyzer.get_numeric_statistics(df)
    if not numeric_stats.empty:
        st.dataframe(numeric_stats, use_container_width=True)

def show_threshold_analysis():
    """Показать вкладку порогового анализа"""
    st.header("🌊 Пороговый анализ уровня воды")
    
    # Информация о сезонности
    st.info("""
    **ℹ️ Внимание:** Анализ выполняется только для сезона открытой воды (май-октябрь).  
    Зимние месяцы (ноябрь-апрель) не анализируются, так как река замерзает и данные отсутствуют.
    """)
    
    # Проверяем наличие предсказателя
    if 'predictor' not in st.session_state:
        st.warning("Сначала обучите модель на вкладке 'Предсказания'")
        return
    
    # Получаем данные
    df = None
    date_columns = []
    
    if 'current_df' in st.session_state:
        df = st.session_state['current_df']
    elif 'special_coords_df' in st.session_state:
        df = st.session_state['special_coords_df']
    
    if df is None or df.empty:
        st.warning("Нет данных для анализа")
        return
    
    # Определяем колонки с датами
    date_columns = st.session_state.predictor.detect_date_columns(df)
    if not date_columns:
        st.warning("Не найдены колонки с датами")
        return
    
    # Фильтруем данные только по сезону открытой воды (май-октябрь)
    df_filtered = df.copy()
    for date_col in date_columns:
        try:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
            # Определяем месяцы
            df_filtered['_month'] = df_filtered[date_col].dt.month
            # Оставляем только май-октябрь
            df_filtered = df_filtered[df_filtered['_month'].between(5, 10)]
            break
        except Exception as e:
            continue
    
    if df_filtered.empty:
        st.warning("Нет данных за сезон открытой воды (май-октябрь)")
        return
    
    # Определяем потенциальные колонки с уровнем воды
    water_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['уровен', 'water', 'вод', 'level', 'высота', 'глубин']):
            if pd.api.types.is_numeric_dtype(df[col]):
                water_cols.append(col)
    
    if not water_cols:
        st.warning("Не найдены числовые колонки с данными уровня воды")
        st.info("Используйте вкладку 'Корректировка типов' для преобразования данных в числовой формат")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_column = st.selectbox(
            "Выберите колонку с датами:",
            date_columns,
            key="threshold_date_col"
        )
    
    with col2:
        water_column = st.selectbox(
            "Выберите колонку уровня воды:",
            water_cols,
            key="threshold_water_col"
        )
    
    with col3:
        # Информация о данных
        if water_column in df_filtered.columns:
            water_data = df_filtered[water_column].dropna()
            st.metric(
                "Данных за сезон",
                f"{len(water_data)} записей",
                help="Только май-октябрь"
            )
    
    st.markdown("---")
    
    # Настройка порогов
    st.subheader("⚙️ Настройка пороговых значений")
    
    if water_column in df_filtered.columns:
        water_data = df_filtered[water_column].dropna()
        
        if len(water_data) > 0:
            # Статистика для автоопределения (только по сезонным данным!)
            auto_high = float(water_data.quantile(0.85))
            auto_danger = float(water_data.quantile(0.95))
            min_val = float(water_data.min())
            max_val = float(water_data.max())
            mean_val = float(water_data.mean())
            
            # Чтение справочника ГМС порогов
            gms_file = Path("data/гмс_карта.csv")
            gms_list = ["(Автоматический расчет)"]
            gms_data = None
            if gms_file.exists():
                try:
                    gms_data = pd.read_csv(gms_file)
                    if 'ГМС' in gms_data.columns:
                        gms_list.extend(gms_data['ГМС'].dropna().astype(str).tolist())
                except:
                    pass
            
            # Выбор гидропоста для автоподстановки порогов
            selected_gms = st.selectbox("📌 Выберите гидропост для загрузки оф. порогов ОЯ (Опасных Явлений):", gms_list)
            
            if selected_gms != "(Автоматический расчет)" and gms_data is not None:
                row = gms_data[gms_data['ГМС'].astype(str) == selected_gms].iloc[0]
                try:
                    # Пытаемся распарсить числа (могут быть с запятой или нечисловыми)
                    danger_str = str(row.get('Крит  уровень ОЯ', '')).replace(',', '.')
                    high_str = str(row.get('Низкий уровень ОЯ', '')).replace(',', '.')
                    
                    if danger_str and danger_str != 'nan':
                        auto_danger = float(danger_str)
                    if high_str and high_str != 'nan':
                        auto_high = float(high_str)
                        
                    st.success(f"Загружены официальные пороги гидропоста: {selected_gms}")
                except Exception as e:
                    st.warning(f"Не удалось распознать уровни для станции {selected_gms}")
            
            # Анализ сезонности данных
            months_data = {}
            for month in range(5, 11):  # Май-октябрь
                month_data = df_filtered[df_filtered['_month'] == month][water_column].dropna()
                if len(month_data) > 0:
                    months_data[month] = {
                        'name': ['Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь'][month-5],
                        'mean': float(month_data.mean()),
                        'min': float(month_data.min()),
                        'max': float(month_data.max())
                    }
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.info("**📊 Статистика по сезону открытой воды:**")
                st.write(f"Минимум: {min_val:.2f}")
                st.write(f"Среднее: {mean_val:.2f}")
                st.write(f"Максимум: {max_val:.2f}")
                st.write(f"85-й перцентиль: {auto_high:.2f}")
                st.write(f"95-й перцентиль: {auto_danger:.2f}")
                st.write(f"Записей: {len(water_data)}")
            
            with col_stat2:
                # Статистика по месяцам
                if months_data:
                    st.info("**📅 Средние значения по месяцам:**")
                    for month, data in months_data.items():
                        st.write(f"{data['name']}: {data['mean']:.2f} (от {data['min']:.2f} до {data['max']:.2f})")
                
                # Настройка порогов
                high_threshold = st.number_input(
                    "Порог 'Высокий уровень':",
                    value=auto_high,
                    step=0.1,
                    help="Уровень, при котором начинается усиленный мониторинг"
                )
                
                danger_threshold = st.number_input(
                    "Порог 'Опасный уровень':",
                    value=auto_danger,
                    step=0.1,
                    help="Уровень, требующий экстренных мер"
                )
                
                # Установка порогов
                if st.button("✅ Установить пороги", key="set_thresholds_btn"):
                    try:
                        if hasattr(st.session_state.predictor, 'set_thresholds'):
                            st.session_state.predictor.set_thresholds(high_threshold, danger_threshold)
                            st.success(f"Пороги установлены: Высокий={high_threshold}, Опасный={danger_threshold}")
                    except Exception as e:
                        st.error(f"Ошибка установки порогов: {str(e)}")
    
    st.markdown("---")
    
    # Кнопка запуска анализа
    if st.button("🌊 Запустить пороговый анализ", type="primary", key="run_threshold_analysis"):
        if not hasattr(st.session_state.predictor, 'thresholds_set') or not st.session_state.predictor.thresholds_set:
            st.error("Сначала установите пороговые значения!")
            return
        
        # Проверяем, обучена ли модель
        if water_column not in st.session_state.predictor.models:
            st.warning("Модель не обучена. Обучаю на сезонных данных...")
            with st.spinner("Обучение модели на данных май-октябрь..."):
                try:
                    # Используем специальный метод для обучения на сезонных данных
                    df_for_training = df_filtered.drop(columns=['_month'], errors='ignore')
                    score = st.session_state.predictor.train_model(df_for_training, water_column, date_column)
                    st.success(f"Модель обучена на сезонных данных! R² = {score:.3f}")
                except Exception as e:
                    st.error(f"Ошибка обучения модели: {str(e)}")
                    return
        
        with st.spinner("Выполняю пороговый анализ для сезона открытой воды..."):
            try:
                # Получаем анализ с порогами (используем сезонный метод)
                report = st.session_state.predictor.analyze_with_thresholds(
                    df, water_column, date_column
                )
                
                # Сохраняем отчет в session_state
                st.session_state.threshold_report = report
                
                # Отображаем результаты
                display_threshold_results(report, water_column)
                
            except Exception as e:
                st.error(f"Ошибка анализа: {str(e)}")
    
    # Если есть сохраненный отчет, показываем его
    if 'threshold_report' in st.session_state:
        st.markdown("---")
        if st.button("🔄 Обновить отчет", key="refresh_threshold_report"):
            del st.session_state.threshold_report
            st.rerun()
        else:
            display_threshold_results(st.session_state.threshold_report, water_column)

def unique_key():
    """
    Возвращает уникальный ключ каждый раз при вызове
    Использование: key=unique_key()
    """
    return f"key_{time.time_ns()}_{random.randint(10000, 99999)}"

def display_threshold_results(report, water_column):
    """Отображение результатов порогового анализа"""
    
    if not report:
        st.warning("Нет данных для отображения")
        return
    
    # Добавляем информацию о сезонности
    st.info(f"""
    **📅 Сезон открытой воды:** {report.get('season_period', 'май-октябрь')}  
    **Год прогноза:** {report.get('year', 'N/A')}  
    **Примечание:** {report.get('note', 'Прогноз только для сезона открытой воды')}
    """)
    
    analysis = report.get('threshold_analysis', {})
    classification = report.get('danger_classification', [])
    high_threshold = report.get('high_threshold', 0)
    danger_threshold = report.get('danger_threshold', 0)
    
    # 1. Сводка
    st.subheader("📊 Сводка по порогам")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_days = len(analysis.get('days_above_high', []))
        st.metric(
            label="Дней выше 'Высокого'", 
            value=high_days,
            delta=f"Макс: {analysis.get('max_exceedance_high', 0):.2f} м"
        )
    
    with col2:
        danger_days = len(analysis.get('days_above_danger', []))
        st.metric(
            label="Дней выше 'Опасного'", 
            value=danger_days,
            delta=f"Макс: {analysis.get('max_exceedance_danger', 0):.2f} м"
        )
    
    with col3:
        high_duration = analysis.get('high_period_duration', 0)
        st.metric(
            label="Период 'Высокого'", 
            value=f"{high_duration} дн.",
            delta=f"Макс: {analysis.get('longest_high_period', 0)} дн."
        )
    
    with col4:
        danger_duration = analysis.get('danger_period_duration', 0)
        st.metric(
            label="Период 'Опасного'", 
            value=f"{danger_duration} дн." if danger_duration > 0 else "Нет"
        )
    
    # 2. Даты превышений
    st.subheader("📅 Ключевые даты")
    
    col_date1, col_date2 = st.columns(2)
    
    with col_date1:
        first_high = analysis.get('first_high_date')
        if first_high:
            st.info(f"**Первое превышение 'Высокого':** {first_high.strftime('%d.%m.%Y')}")
        else:
            st.info("**Первое превышение 'Высокого':** Не ожидается")
        
        last_high = analysis.get('last_high_date')
        if last_high:
            st.info(f"**Последнее превышение 'Высокого':** {last_high.strftime('%d.%m.%Y')}")
    
    with col_date2:
        first_danger = analysis.get('first_danger_date')
        if first_danger:
            st.error(f"**Первое превышение 'Опасного':** {first_danger.strftime('%d.%m.%Y')}")
        else:
            st.success("**Первое превышение 'Опасного':** Не ожидается")
        
        last_danger = analysis.get('last_danger_date')
        if last_danger:
            st.error(f"**Последнее превышение 'Опасного':** {last_danger.strftime('%d.%m.%Y')}")
    
    # 3. График с порогами
    st.subheader("📈 Визуализация с порогами")
    
    # Создаем DataFrame для графика
    dates = [day['date'] for day in classification]
    values = [day['value'] for day in classification]
    colors = [day['color'] for day in classification]
    levels = [day['level'] for day in classification]
    
    # Создаем график
    fig = go.Figure()
    
    # Добавляем прогноз
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines',
        name='Прогноз уровня',
        line=dict(color='blue', width=2),
        hovertemplate='%{x|%d.%m.%Y}<br>Уровень: %{y:.2f} м<extra></extra>'
    ))
    
    # Добавляем пороговые линии
    fig.add_hline(
        y=high_threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text="Высокий уровень",
        annotation_position="bottom right",
        annotation_font_size=12
    )
    
    fig.add_hline(
        y=danger_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Опасный уровень",
        annotation_position="bottom right",
        annotation_font_size=12
    )
    
    # Добавляем цветные маркеры по опасности
    danger_points = {'x': [], 'y': [], 'color': []}
    warning_points = {'x': [], 'y': [], 'color': []}
    normal_points = {'x': [], 'y': [], 'color': []}
    
    for i, (date, value, color, level) in enumerate(zip(dates, values, colors, levels)):
        if level == 'danger':
            danger_points['x'].append(date)
            danger_points['y'].append(value)
            danger_points['color'].append(color)
        elif level == 'warning':
            warning_points['x'].append(date)
            warning_points['y'].append(value)
            warning_points['color'].append(color)
        else:
            normal_points['x'].append(date)
            normal_points['y'].append(value)
            normal_points['color'].append(color)
    
    # Добавляем точки
    if danger_points['x']:
        fig.add_trace(go.Scatter(
            x=danger_points['x'], y=danger_points['y'],
            mode='markers',
            name='Опасный уровень',
            marker=dict(color='red', size=8, symbol='diamond'),
            hovertemplate='%{x|%d.%m.%Y}<br>Уровень: %{y:.2f} м<br>Статус: 🔴 Опасный<extra></extra>'
        ))
    
    if warning_points['x']:
        fig.add_trace(go.Scatter(
            x=warning_points['x'], y=warning_points['y'],
            mode='markers',
            name='Высокий уровень',
            marker=dict(color='orange', size=6, symbol='square'),
            hovertemplate='%{x|%d.%m.%Y}<br>Уровень: %{y:.2f} м<br>Статус: 🟡 Высокий<extra></extra>'
        ))
    
    if normal_points['x']:
        fig.add_trace(go.Scatter(
            x=normal_points['x'], y=normal_points['y'],
            mode='markers',
            name='Нормальный уровень',
            marker=dict(color='green', size=4, symbol='circle'),
            hovertemplate='%{x|%d.%m.%Y}<br>Уровень: %{y:.2f} м<br>Статус: 🟢 Норма<extra></extra>'
        ))
    
    # Настройки графика
    fig.update_layout(
        title=f"Прогноз уровня воды с порогами опасности ({water_column})",
        xaxis_title="Дата",
        yaxis_title="Уровень, м",
        hovermode="x unified",
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Добавляем уникальный ключ к графику
    datetime.now().strftime("%Y%m%d_%H%M%S")
    st.plotly_chart(fig, use_container_width=True, key=unique_key())
    
    # 4. Таблица опасных дней
    danger_days_data = [day for day in classification if day['level'] != 'normal']
    
    if danger_days_data:
        st.subheader("⚠️ Дни с превышением порогов")
        
        # Создаем DataFrame для таблицы
        danger_df = pd.DataFrame([
            {
                'Дата': day['formatted_date'],
                'Уровень': f"{day['value']:.2f} м",
                'Статус': day['icon'] + {
                    'danger': ' Опасный',
                    'warning': ' Высокий'
                }[day['level']],
                'Превышение высокого': f"{day['exceedance_high']:.2f} м" if day['exceedance_high'] is not None else "-",
                'Превышение опасного': f"{day['exceedance_danger']:.2f} м" if day['exceedance_danger'] is not None else "-"
            }
            for day in danger_days_data
        ])
        
        st.dataframe(
            danger_df,
            use_container_width=True,
            hide_index=True,
            key=unique_key()
        )
        
        # Экспорт данных
        csv_data = danger_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Экспорт опасных дней (CSV)",
            data=csv_data,
            file_name="danger_days.csv",
            mime="text/csv",
            type="secondary",
            key=unique_key()
        )
    else:
        st.success("✅ Дней с превышением порогов не обнаружено")
    
    # 5. Рекомендации
    st.subheader("💡 Рекомендации")
    
    if hasattr(st.session_state.predictor, 'generate_threshold_recommendations'):
        recommendations = st.session_state.predictor.generate_threshold_recommendations(analysis)
        
        for i, rec in enumerate(recommendations):
            with st.container():
                if rec['type'] == 'critical':
                    st.error(rec['text'], icon="⚠️")
                elif rec['type'] == 'warning':
                    st.warning(rec['text'], icon="🟡")
                elif rec['type'] == 'info':
                    st.info(rec['text'], icon="ℹ️")
                else:
                    st.success(rec['text'], icon="✅")
                
                # Действия
                with st.expander(f"Рекомендуемые действия ({i+1})"):
                    for action in rec.get('actions', []):
                        st.markdown(f"- {action}")
                    
                    # Дополнительные детали
                    if 'details' in rec:
                        st.markdown("**Детали:**")
                        for detail in rec['details']:
                            st.markdown(f"- {detail}")
                    
                    # Периоды если есть
                    if 'periods' in rec:
                        st.markdown("**Опасные периоды:**")
                        for period in rec['periods']:
                            st.markdown(f"- {period}")
    else:
        # Базовые рекомендации
        st.info("""
        **Базовые рекомендации по анализу уровня воды:**
        
        1. **При превышении высокого уровня:** Усильте мониторинг, проверьте готовность оборудования
        2. **При превышении опасного уровня:** Активируйте систему оповещения, подготовьте защитные сооружения
        3. **В нормальный период:** Продолжайте регулярные наблюдения, обновляйте данные
        """, icon="ℹ️")
    
    # 6. Экспорт отчета
    st.markdown("---")
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("💾 Сохранить полный отчет", type="primary", key="save_full_report_btn"):
            # Создаем текст отчета
            report_text = f"""
            ОТЧЕТ ПОРОГОВОГО АНАЛИЗА
            ========================
            
            Колонка уровня воды: {water_column}
            Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
            
            ПОРОГИ:
            - Высокий уровень: {high_threshold} м
            - Опасный уровень: {danger_threshold} м
            
            СВОДКА:
            - Дней выше высокого уровня: {len(analysis.get('days_above_high', []))}
            - Дней выше опасного уровня: {len(analysis.get('days_above_danger', []))}
            - Максимальное превышение опасного уровня: {analysis.get('max_exceedance_danger', 0):.2f} м
            
            КЛЮЧЕВЫЕ ДАТЫ:
            - Первое превышение высокого: {analysis.get('first_high_date', 'Не ожидается')}
            - Последнее превышение высокого: {analysis.get('last_high_date', 'Нет')}
            - Первое превышение опасного: {analysis.get('first_danger_date', 'Не ожидается')}
            - Последнее превышение опасного: {analysis.get('last_danger_date', 'Нет')}
            
            СЕЗОН ОТКРЫТОЙ ВОДЫ:
            - Период: {report.get('season_period', 'май-октябрь')}
            - Год: {report.get('year', 'N/A')}
            - Примечание: {report.get('note', 'Прогноз только для сезона открытой воды')}
            """
            
            # Сохраняем файл
            report_path = Path("reports") / "threshold_analysis" / f"пороговый_анализ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            st.success(f"Отчет сохранен: {report_path}")
    
    with col_export2:
        # Кнопка для экспорта графика
        if st.button("📊 Экспорт графика", type="secondary", key="save_grav_report_button"):
            # Сохраняем график как HTML
            chart_path = Path("reports") / "charts" / f"график_пороги_{water_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            chart_path.parent.mkdir(exist_ok=True, parents=True)
            
            fig.write_html(str(chart_path))
            st.success(f"График сохранен: {chart_path}")
            
def show_data_type_correction():
    """Показать инструмент для ручной корректировки типов данных"""
    if 'current_df' not in st.session_state:
        st.info("Сначала загрузите данные")
        return
    
    df = st.session_state['current_df']
    
    st.header("🛠️ Корректировка типов данных")
    st.info("Если система неправильно определила типы данных, вы можете исправить их вручную")
    
    # Выбор колонки для корректировки
    selected_col = st.selectbox("Выберите колонку для корректировки:", df.columns, key="type_correction_col")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Текущий тип")
        current_type = str(df[selected_col].dtype)
        st.info(f"**{current_type}**")
        
        # Показываем примеры значений
        st.write("Примеры значений:")
        sample_values = df[selected_col].dropna().head(5).tolist()
        for val in sample_values:
            st.code(f"{val}")
    
    with col2:
        st.subheader("Новый тип")
        new_type = st.selectbox(
            "Выберите новый тип:",
            ["Автоопределение", "Числовой (float)", "Целочисленный (int)", 
             "Дата/время (datetime)", "Текст (string)", "Категория (category)"],
            key=f"type_select_{selected_col}"
        )
    
    if st.button("🔧 Применить преобразование", key=f"apply_conversion_{selected_col}"):
        with st.spinner("Преобразую данные..."):
            df_modified = df.copy()
            
            try:
                if new_type == "Числовой (float)":
                    df_modified[selected_col] = pd.to_numeric(df_modified[selected_col], errors='coerce')
                    st.success(f"Колонка '{selected_col}' преобразована в числовой формат")
                
                elif new_type == "Целочисленный (int)":
                    df_modified[selected_col] = pd.to_numeric(df_modified[selected_col], errors='coerce')
                    df_modified[selected_col] = df_modified[selected_col].astype('Int64')
                    st.success(f"Колонка '{selected_col}' преобразована в целочисленный формат")
                
                elif new_type == "Дата/время (datetime)":
                    df_modified[selected_col] = pd.to_datetime(df_modified[selected_col], errors='coerce', format='mixed')
                    st.success(f"Колонка '{selected_col}' преобразована в формат даты/времени")
                
                elif new_type == "Текст (string)":
                    df_modified[selected_col] = df_modified[selected_col].astype(str)
                    st.success(f"Колонка '{selected_col}' преобразована в текстовый формат")
                
                elif new_type == "Категория (category)":
                    df_modified[selected_col] = df_modified[selected_col].astype('category')
                    st.success(f"Колонка '{selected_col}' преобразована в категориальный формат")
                
                elif new_type == "Автоопределение":
                    # Используем метод detect_and_convert_dtypes если он есть
                    if hasattr(st.session_state.analyzer, 'detect_and_convert_dtypes'):
                        df_modified = st.session_state.analyzer.detect_and_convert_dtypes(df_modified)
                    else:
                        # Просто пробуем определить тип автоматически
                        # Пробуем дату
                        date_test = pd.to_datetime(df_modified[selected_col], errors='coerce', format='mixed')
                        if date_test.notna().any():
                            df_modified[selected_col] = date_test
                        else:
                            # Пробуем число
                            numeric_test = pd.to_numeric(df_modified[selected_col], errors='coerce')
                            if numeric_test.notna().any():
                                df_modified[selected_col] = numeric_test
                    
                    st.success("Типы данных определены автоматически")
                
                # Сохраняем изменения
                st.session_state['current_df'] = df_modified
                
                # Показываем результат
                st.subheader("Результат преобразования")
                st.dataframe(df_modified[[selected_col]].head(10), use_container_width=True)
                
                # Предлагаем обновить интерфейс
                if st.button("🔄 Обновить интерфейс с новыми данными", key="reload_after_conversion"):
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Ошибка преобразования: {str(e)}")

def show_visualization():
    """Показать вкладку визуализации"""
    st.header("📊 Интерактивная визуализация")
    
    # Получаем данные
    has_coords = st.session_state.get('has_coords', False)
    df = None
    
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        if not gdf.empty:
            df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    elif 'current_df' in st.session_state:
        df = st.session_state['current_df']
    
    if df is None or df.empty:
        st.warning("Нет данных для визуализации")
        return
    
    # Получаем список доступных типов графиков
    plot_types_dict = st.session_state.visualizer.get_plot_types()
    
    # Создаем плоский список типов графиков для выбора
    flat_plot_types = []
    plot_type_categories = {}
    
    for category, plots in plot_types_dict.items():
        for plot_type in plots:
            flat_plot_types.append(f"{category}: {plot_type}")
            plot_type_categories[plot_type] = category
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_plot_display = st.selectbox(
            "Тип графика",
            flat_plot_types,
            key="viz_type"
        )
        
        # Извлекаем чистый тип графика
        plot_type = selected_plot_display.split(": ")[1] if ": " in selected_plot_display else selected_plot_display
        
        x_col = st.selectbox("Ось X", df.columns.tolist(), key="viz_x")
        
        if plot_type not in ["histogram", "pie", "density"]:
            y_col = st.selectbox("Ось Y", df.columns.tolist(), key="viz_y")
        else:
            y_col = None
    
    with col2:
        # Выбор цветовой палитры
        palettes = st.session_state.visualizer.get_available_color_palettes()
        color_palette = st.selectbox("Цветовая палитра", palettes, key="viz_palette")
        
        # Дополнительные настройки в зависимости от типа графика
        if plot_type in ["scatter", "line", "bar", "box", "violin"]:
            color_options = ["Нет"] + df.columns.tolist()
            color_col = st.selectbox("Цвет", color_options, key="viz_color")
            if color_col == "Нет":
                color_col = None
        else:
            color_col = None
        
        plot_title = st.text_input("Название графика", value=f"{plot_type.capitalize()} график", key="viz_title")
    
    # Настройки подсказок
    with st.expander("💡 Настройки подсказок"):
        col_hover1, col_hover2 = st.columns(2)
        with col_hover1:
            hover_bg_color = st.color_picker("Цвет фона", "#FFFFFF", key="hover_bg")
            hover_font_size = st.slider("Размер шрифта", 8, 16, 12, key="hover_font")
        with col_hover2:
            hover_text_color = st.color_picker("Цвет текста", "#000000", key="hover_text")
            hover_border_color = st.color_picker("Цвет границы", "#CCCCCC", key="hover_border")
    
    if st.button("🔄 Создать график", type="primary", key="create_viz_btn"):
        try:
            fig = None
            
            # Создание графиков в зависимости от типа
            if plot_type == "scatter":
                fig = st.session_state.visualizer.create_scatter_plot(
                    df, x_col, y_col, color_col, 
                    title=plot_title,
                    color_scale=color_palette
                )
            
            elif plot_type == "line":
                fig = st.session_state.visualizer.create_line_plot(
                    df, x_col, [y_col], color_col,
                    title=plot_title,
                    color_palette=color_palette
                )
            
            elif plot_type == "bar":
                fig = st.session_state.visualizer.create_bar_chart(
                    df, x_col, y_col, color_col,
                    title=plot_title,
                    color_palette=color_palette
                )
            
            elif plot_type == "histogram":
                fig = st.session_state.visualizer.create_histogram(
                    df, x_col, color_col,
                    title=plot_title,
                    color_palette=color_palette
                )
            
            elif plot_type == "pie":
                fig = st.session_state.visualizer.create_pie_chart(
                    df, x_col,
                    title=plot_title,
                    color_palette=color_palette
                )
            
            elif plot_type == "box":
                fig = st.session_state.visualizer.create_box_plot(
                    df, x_col, y_col, color_col,
                    title=plot_title,
                    color_palette=color_palette
                )
            
            elif plot_type == "violin":
                fig = st.session_state.visualizer.create_violin_plot(
                    df, x_col, y_col, color_col,
                    title=plot_title,
                    color_palette=color_palette
                )
            
            elif plot_type == "density":
                # Выбор колонок для плотности
                density_cols = st.multiselect("Выберите колонки", 
                                            df.select_dtypes(include=[np.number]).columns.tolist(),
                                            default=[df.select_dtypes(include=[np.number]).columns[0]])
                if density_cols:
                    fig = st.session_state.visualizer.create_density_plot(
                        df, density_cols,
                        title=plot_title,
                        color_palette=color_palette
                    )
            
            elif plot_type == "correlation_matrix":
                fig = st.session_state.visualizer.create_correlation_matrix(
                    df,
                    title=plot_title,
                    color_scale=color_palette
                )
            
            elif plot_type == "scatter_matrix":
                dimensions = st.multiselect("Выберите колонки для матрицы", 
                                          df.select_dtypes(include=[np.number]).columns.tolist(),
                                          default=df.select_dtypes(include=[np.number]).columns[:5])
                if len(dimensions) >= 2:
                    fig = st.session_state.visualizer.create_scatter_matrix(
                        df, dimensions, color_col,
                        title=plot_title,
                        color_palette=color_palette
                    )
            
            elif plot_type == "3d_scatter":
                z_options = df.select_dtypes(include=[np.number]).columns.tolist()
                z_col = st.selectbox("Ось Z", z_options, key="viz_z")
                
                fig = st.session_state.visualizer.create_3d_scatter(
                    df, x_col, y_col, z_col, color_col,
                    title=plot_title,
                    color_scale=color_palette
                )
            
            elif plot_type == "time_series":
                # Выбор временной колонки
                time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if time_cols:
                    time_col = st.selectbox("Временная колонка", time_cols, key="viz_time")
                    value_cols = st.multiselect("Значения для отображения", 
                                              df.select_dtypes(include=[np.number]).columns.tolist(),
                                              default=df.select_dtypes(include=[np.number]).columns[:3])
                    if value_cols:
                        fig = st.session_state.visualizer.create_time_series(
                            df, time_col, value_cols,
                            title=plot_title,
                            color_palette=color_palette
                        )
            
            if fig is not None:
                # Применяем настройки подсказок
                fig.update_layout(
                    hoverlabel=dict(
                        bgcolor=hover_bg_color,
                        font_size=hover_font_size,
                        font_color=hover_text_color,
                        bordercolor=hover_border_color
                    )
                )
                
                # Отображение графика
                st.plotly_chart(fig, use_container_width=True, key=unique_key())
                
                # Опции сохранения
                with st.expander("💾 Опции сохранения"):
                    col_save1, col_save2, col_save3 = st.columns(3)
                    
                    with col_save1:
                        if st.button("Сохранить как HTML", key="save_html_btn"):
                            saved_files = st.session_state.visualizer.save_plot(
                                fig, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                formats=['html']
                            )
                            if saved_files:
                                st.success(f"Сохранено: {saved_files[0]}")
                    
                    with col_save2:
                        if st.button("Сохранить как PNG", key="save_png_btn"):
                            saved_files = st.session_state.visualizer.save_plot(
                                fig, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                formats=['png']
                            )
                            if saved_files:
                                st.success(f"Сохранено: {saved_files[0]}")
                    
                    with col_save3:
                        if st.button("Сохранить все форматы", key="save_all_btn"):
                            saved_files = st.session_state.visualizer.save_plot(
                                fig, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                formats=['html', 'png', 'json']
                            )
                            if saved_files:
                                st.success(f"Сохранено {len(saved_files)} файлов")
                
        except Exception as e:
            st.error(f"Ошибка создания графика: {e}")

def show_map_view():
    """Показать вкладку карты с поддержкой русских карт"""
    st.header("🗺️ Интерактивная карта")
    
    # Импортируем streamlit_folium здесь, чтобы избежать конфликтов
    from streamlit_folium import st_folium
    
    # Проверяем наличие геоданных
    gdf = st.session_state.get('current_gdf')
    
    # Если нет геоданных, но есть обычный DataFrame с координатами
    if gdf is None or gdf.empty:
        if 'current_df' in st.session_state:
            df = st.session_state['current_df']
            
            # Проверяем, есть ли координаты в DataFrame
            lat_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['широт', 'lat', 'latitude'])]
            lon_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['долгот', 'lon', 'longitude'])]
            
            if lat_cols and lon_cols:
                try:
                    # Преобразуем координаты из вашего формата N56.77882594,E105.75483468
                    def parse_coordinate(coord):
                        """Парсит координаты из формата N56.77882594 или E105.75483468"""
                        if pd.isna(coord):
                            return None
                        
                        coord_str = str(coord).strip()
                        if not coord_str:
                            return None
                        
                        # Если координата уже числовая
                        if isinstance(coord, (int, float)):
                            return float(coord)
                        
                        # Убираем буквы N, E, S, W
                        coord_str = coord_str.upper().replace('N', '').replace('E', '').replace('S', '').replace('W', '').replace(',', '')
                        
                        try:
                            return float(coord_str)
                        except Exception as e:
                            return None
                    
                    # Создаем копию DataFrame с преобразованными координатами
                    df_coords = df.copy()
                    df_coords['latitude_parsed'] = df_coords[lat_cols[0]].apply(parse_coordinate)
                    df_coords['longitude_parsed'] = df_coords[lon_cols[0]].apply(parse_coordinate)
                    
                    # Удаляем строки с некорректными координатами
                    df_coords = df_coords.dropna(subset=['latitude_parsed', 'longitude_parsed'])
                    
                    if not df_coords.empty:
                        # Создаем геоданные
                        from shapely.geometry import Point
                        import geopandas as gpd
                        
                        geometry = [Point(lon, lat) for lon, lat in zip(df_coords['longitude_parsed'], df_coords['latitude_parsed'])]
                        gdf = gpd.GeoDataFrame(df_coords, geometry=geometry, crs="EPSG:4326")
                        st.session_state['current_gdf'] = gdf
                        st.success(f"✅ Созданы геоданные из координат ({len(gdf)} точек)")
                        st.rerun()  # Перезагружаем для обновления интерфейса
                    else:
                        st.warning("Не удалось преобразовать координаты")
                        return
                except Exception as e:
                    st.error(f"Ошибка создания геоданных: {e}")
                    return
            else:
                st.warning("В данных не найдены колонки с координатами")
                st.info("Ищите колонки с названиями, содержащими: широта, долгота, lat, lon, latitude, longitude")
                return
        else:
            st.warning("Нет данных для отображения")
            return
    
    # Теперь у нас точно есть gdf
    df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    
    # Статистика по данным
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Всего точек", len(gdf))
    with col_stat2:
        bounds = gdf.total_bounds if hasattr(gdf, 'total_bounds') else None
        if bounds is not None:
            st.metric("Широта", f"{bounds[1]:.4f} - {bounds[3]:.4f}")
    with col_stat3:
        if bounds is not None:
            st.metric("Долгота", f"{bounds[0]:.4f} - {bounds[2]:.4f}")
    
    # Основные настройки карты
    col1, col2 = st.columns(2)
    
    with col1:
        # Выбор типа карты - включаем русские карты
        map_types = {
            "OpenStreetMap": "OpenStreetMap",
            "Карты 2ГИС": "2gis", 
            "Яндекс.Карты (схема)": "yandex_map",
            "Яндекс.Карты (спутник)": "yandex_sat",
            "Google Maps (схема)": "google_map",
            "Google Maps (спутник)": "google_sat",
            "Stamen Terrain": "Stamen Terrain",
            "CartoDB positron": "CartoDB positron"
        }
        
        map_type_display = list(map_types.keys())
        selected_map = st.selectbox(
            "Тип карты",
            map_type_display,
            key="map_type_select"
        )
        map_types[selected_map]
        
        # Выбор колонки для цвета
        color_options = ["Нет (синий)"] + df.columns.tolist()
        color_by = st.selectbox("Цвет по колонке", color_options, key="map_color_select")
        
        # Настройка радиуса маркеров
        marker_radius = st.slider("Радиус маркера", 3, 20, 6, key="radius_slider")
    
    with col2:
        # Выбор колонки для размера
        size_options = ["Нет (фикс.)"] + df.select_dtypes(include=[np.number]).columns.tolist()
        size_by = st.selectbox("Размер по колонке", size_options, key="map_size_select")
        
        # Выбор колонок для подсказки
        popup_options = df.columns.tolist()
        popup_cols = st.multiselect(
            "Колонки в подсказке", 
            popup_options, 
            default=popup_options[:3] if len(popup_options) > 3 else popup_options,
            key="popup_cols_select"
        )
        
        # Кластеризация маркеров
        use_clustering = st.checkbox("Кластеризация маркеров", value=len(gdf) > 50, key="clustering_checkbox")
        
        # Масштабирование для России
        if st.checkbox("Оптимизировать для России", value=True, key="optimize_russia"):
            st.info("Увеличит зум для лучшего отображения России")
    
    # Кнопка для обновления карты с выбранными параметрами
    if st.button("🔄 Обновить карту с новыми параметрами", key="update_map_btn"):
        st.session_state['map_updated'] = True
        st.rerun()
    
    # Автоматическое создание карты
    try:
        # Вычисляем средние координаты
        if hasattr(gdf, 'geometry'):
            # Вычисляем средние координаты
            avg_lat = gdf.geometry.y.mean() if hasattr(gdf.geometry, 'y') else 0
            avg_lon = gdf.geometry.x.mean() if hasattr(gdf.geometry, 'x') else 0
        else:
            avg_lat, avg_lon = 62, 129  # Центр Якутии
        
        # Автоматически определяем подходящий зум
        if hasattr(gdf, 'total_bounds') and gdf.total_bounds is not None:
            bounds = gdf.total_bounds
            lat_range = bounds[3] - bounds[1]
            if lat_range < 1:
                zoom_start = 10
            elif lat_range < 5:
                zoom_start = 7
            elif lat_range < 20:
                zoom_start = 5
            else:
                zoom_start = 4
        else:
            zoom_start = 5
        
        # Корректировка зума для России
        if st.session_state.get('optimize_russia', True) and avg_lat > 50:
            zoom_start = max(zoom_start, 4)
        
        # Управление офлайн/онлайн режимом
        map_bg_mode = st.radio(
            "Режим отображения карты:",
            ["Онлайн (С подложкой OSM/Яндекс)", "Офлайн (Только геометрия, без интернета)"],
            horizontal=True
        )
        
        base_tiles = 'OpenStreetMap' if map_bg_mode.startswith("Онлайн") else None

        # Создаем базовую карту
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=zoom_start,
            tiles=base_tiles,  # Базовый слой
            control_scale=True,
            width='100%',
            height=600,
        )
        
        # Добавляем различные типы карт как слои
        # Яндекс.Карты
        yandex_map_tiles = 'https://vec01.maps.yandex.net/tiles?l=map&x={x}&y={y}&z={z}'
        yandex_sat_tiles = 'https://vec01.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}'
        
        # 2ГИС
        dgis_tiles = 'https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}'
        
        # Google Maps
        google_map_tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
        google_sat_tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        
        if base_tiles is not None:
            # Добавляем слои карт
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='OpenStreetMap',
                attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                control=True
            ).add_to(m)
            
            # Яндекс.Карты (схема)
            folium.TileLayer(
                tiles=yandex_map_tiles,
                name='Яндекс.Карты (схема)',
                attr='© <a href="https://yandex.ru/maps/">Яндекс.Карты</a>',
                control=True
            ).add_to(m)
            
            # Яндекс.Карты (спутник)
            folium.TileLayer(
                tiles=yandex_sat_tiles,
                name='Яндекс.Карты (спутник)',
                attr='© <a href="https://yandex.ru/maps/">Яндекс.Карты</a>',
                control=True
            ).add_to(m)
            
            # 2ГИС
            folium.TileLayer(
                tiles=dgis_tiles,
                name='Карты 2ГИС',
                attr='© <a href="https://2gis.ru">2ГИС</a>',
                control=True
            ).add_to(m)
            
            # Google Maps (схема)
            folium.TileLayer(
                tiles=google_map_tiles,
                name='Google Maps (схема)',
                attr='© <a href="https://google.com/maps">Google Maps</a>',
                control=True
            ).add_to(m)
            
            # Google Maps (спутник)
            folium.TileLayer(
                tiles=google_sat_tiles,
                name='Google Maps (спутник)',
                attr='© <a href="https://google.com/maps">Google Maps</a>',
                control=True
            ).add_to(m)
            
            # Stamen Terrain
            folium.TileLayer(
                tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
                name='Stamen Terrain',
                attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
                control=True
            ).add_to(m)
            
            # CartoDB positron
            folium.TileLayer(
                tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                name='CartoDB Positron',
                attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                control=True
            ).add_to(m)
        
        # Добавляем маркеры
        if len(gdf) > 0:
            # Подготовка данных для маркеров
            for idx, row in gdf.iterrows():
                try:
                    if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                        lat, lon = row.geometry.y, row.geometry.x
                    else:
                        continue
                    
                    # Пропускаем некорректные координаты
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Определяем цвет маркера
                    if color_by != "Нет (синий)" and color_by in df.columns:
                        # Используем цветовую палитру для разных значений
                        from branca.colormap import linear
                        
                        if pd.api.types.is_numeric_dtype(df[color_by]):
                            # Числовая колонка - градиентные цвета
                            colormap = linear.YlOrRd_09.scale(
                                df[color_by].min(), 
                                df[color_by].max()
                            )
                            color_val = colormap(row[color_by]) if not pd.isna(row[color_by]) else 'blue'
                        else:
                            # Категориальная колонка - фиксированные цвета
                            unique_vals = df[color_by].dropna().unique()
                            colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 
                                         'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                                         'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                                         'gray', 'black', 'lightgray']
                            
                            # Создаем маппинг значений на цвета
                            color_mapping = {}
                            for i, val in enumerate(unique_vals):
                                color_mapping[val] = colors_list[i % len(colors_list)]
                            
                            color_val = color_mapping.get(row[color_by], 'blue')
                    else:
                        color_val = 'blue'
                    
                    # Определяем размер маркера
                    if size_by != "Нет (фикс.)" and size_by in df.columns and pd.api.types.is_numeric_dtype(df[size_by]):
                        if not pd.isna(row[size_by]):
                            # Нормализуем размер от 3 до 15
                            min_val = df[size_by].min()
                            max_val = df[size_by].max()
                            if max_val > min_val:
                                size = 3 + (row[size_by] - min_val) / (max_val - min_val) * 12
                            else:
                                size = marker_radius
                        else:
                            size = marker_radius
                    else:
                        size = marker_radius
                    
                    # Создаем текст для подсказки
                    popup_text = f"<b>Точка {idx}</b><br>"
                    if popup_cols:
                        for col in popup_cols:
                            if col in df.columns:
                                popup_text += f"<b>{col}:</b> {row[col]}<br>"
                    popup_text += f"<b>Координаты:</b> {lat:.4f}, {lon:.4f}"
                    
                    # Добавляем информацию о реке, если есть
                    if 'Река' in df.columns and not pd.isna(row['Река']):
                        popup_text += f"<br><b>Река:</b> {row['Река']}"
                    if 'ГМС' in df.columns and not pd.isna(row['ГМС']):
                        popup_text += f"<br><b>ГМС:</b> {row['ГМС']}"
                    if 'Район' in df.columns and not pd.isna(row['Район']):
                        popup_text += f"<br><b>Район:</b> {row['Район']}"
                    
                    # Добавляем маркер
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=size,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color_val,
                        fill=True,
                        fill_color=color_val,
                        fill_opacity=0.7,
                        weight=1
                    ).add_to(m)
                    
                except Exception as e:
                    continue
            
            # Если много точек и включена кластеризация
            if use_clustering and len(gdf) > 50:
                from folium.plugins import MarkerCluster
                
                # Создаем кластер
                marker_cluster = MarkerCluster(
                    name="Кластеризованные точки",
                    options={
                        'maxClusterRadius': 50,
                        'showCoverageOnHover': True,
                        'zoomToBoundsOnClick': True,
                        'spiderfyOnMaxZoom': True
                    }
                ).add_to(m)
                
                # Добавляем маркеры в кластер
                for idx, row in gdf.iterrows():
                    try:
                        if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                            lat, lon = row.geometry.y, row.geometry.x
                        else:
                            continue
                        
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                        
                        # Создаем текст для подсказки кластера
                        cluster_popup = f"<b>Точка {idx}</b><br>"
                        if 'ГМС' in df.columns and not pd.isna(row['ГМС']):
                            cluster_popup += f"<b>ГМС:</b> {row['ГМС']}<br>"
                        if 'Река'in df.columns and not pd.isna(row['Река']):
                            cluster_popup += f"<b>Река:</b> {row['Река']}<br>"
                        cluster_popup += f"<b>Координаты:</b> {lat:.4f}, {lon:.4f}"
                        
                        # Простой маркер для кластера
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(cluster_popup, max_width=200),
                            icon=folium.Icon(color='blue', icon='info-sign', prefix='fa')
                        ).add_to(marker_cluster)
                    except Exception as e:
                        continue
        
        # Добавляем мини-карту (инспектор)
        from folium.plugins import MiniMap
        miniMap = MiniMap(position='bottomright')
        m.add_child(miniMap)
        
        # Добавляем полноэкранный режим
        from folium.plugins import Fullscreen
        Fullscreen(position='topright').add_to(m)
        
        # Добавляем измерение расстояний
        from folium.plugins import MeasureControl
        measure = MeasureControl(
            position='topleft',
            primary_length_unit='kilometers',
            secondary_length_unit='meters'
        )
        m.add_child(measure)
        
        # Добавляем слой контроля
        folium.LayerControl(
            position='topright',
            collapsed=True,
            autoZIndex=True
        ).add_to(m)
        
        # Сохраняем карту в session_state
        st.session_state['current_map'] = m
        
        # Отображаем информацию
        st.info("ℹ️ Используйте меню в правом верхнем углу для переключения между картами")
        
        # Отображаем карту с использованием st_folium
        map_data = st_folium(m, width=800, height=600, returned_objects=["last_clicked", "zoom"])
        
        # Показываем информацию о последнем клике
        if map_data and map_data.get("last_clicked"):
            last_click = map_data["last_clicked"]
            st.sidebar.info(f"""
            **Последний клик на карте:**
            - Широта: {last_click['lat']:.6f}
            - Долгота: {last_click['lng']:.6f}
            - Масштаб: {map_data.get('zoom', 'N/A')}
            """)
        
    except Exception as e:
        st.error(f"Ошибка создания карты: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # Опции экспорта
    with st.expander("📤 Опции экспорта"):
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("💾 Сохранить карту как HTML", key="save_map_html_btn"):
                import tempfile
                import os
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"карта_{st.session_state.get('current_file', 'data')}_{timestamp}.html"
                
                # Проверяем наличие карты
                if 'current_map' not in st.session_state:
                    st.warning("Сначала создайте карту")
                    return
                
                # Сохраняем во временный файл
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    st.session_state['current_map'].save(tmp.name)
                    
                    # Читаем и отдаем для скачивания
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    
                    st.download_button(
                        label="📥 Скачать карту",
                        data=map_html,
                        file_name=filename,
                        mime="text/html",
                        key="download_map_button"
                    )
                    
                    os.unlink(tmp.name)
        
        with col_exp2:
            if st.button("🌍 Экспорт данных как GeoJSON", key="export_geojson_btn"):
                if 'current_gdf' in st.session_state:
                    import tempfile
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"данные_{timestamp}.geojson"
                    
                    # Конвертируем в GeoJSON
                    geojson_str = st.session_state['current_gdf'].to_json()
                    
                    st.download_button(
                        label="📥 Скачать GeoJSON",
                        data=geojson_str,
                        file_name=filename,
                        mime="application/json",
                        key="download_geojson_button"
                    )

def show_combined_map(gdf, df):
    """Показать комбинированную карту с переключением слоев"""
    # Создаем базовую карту
    avg_lat = gdf.geometry.y.mean() if hasattr(gdf.geometry, 'y') else 62
    avg_lon = gdf.geometry.x.mean() if hasattr(gdf.geometry, 'x') else 129
    
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=5,
        control_scale=True
        
    )
    
    # Добавляем несколько слоев карт
    # Яндекс.Карты
    yandex_tiles = [
        ('Яндекс.Карты (схема)', 'https://vec01.maps.yandex.net/tiles?l=map&x={x}&y={y}&z={z}', '© Яндекс.Карты'),
        ('Яндекс.Карты (спутник)', 'https://vec01.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}', '© Яндекс.Карты'),
        ('Яндекс.Карты (гибрид)', 'https://vec01.maps.yandex.net/tiles?l=skl&x={x}&y={y}&z={z}', '© Яндекс.Карты')
    ]
    
    # 2ГИС
    dgis_tiles = [
        ('2ГИС (схема)', 'https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}', '© 2ГИС'),
        ('2ГИС (спутник)', 'https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}&v=1', '© 2ГИС')
    ]
    
    # Добавляем все слои
    for name, tiles, attr in yandex_tiles + dgis_tiles:
        folium.TileLayer(
            tiles=tiles,
            name=name,
            attr=attr,
            overlay=True,
            control=True
        ).add_to(m)
    
    # Добавляем маркеры
    for idx, row in gdf.iterrows():
        try:
            lat, lon = row.geometry.y, row.geometry.x
            
            # Создаем информационное окно
            popup_text = "<div style='min-width:250px;'>"
            popup_text += f"<h4>Точка {idx}</h4>"
            
            # Добавляем важные поля
            important_fields = ['ГМС', 'Река', 'Район', 'Наслег', 'Крит  уровень ОЯ', 'Низкий уровень ОЯ']
            for field in important_fields:
                if field in df.columns and not pd.isna(row[field]):
                    popup_text += f"<b>{field}:</b> {row[field]}<br>"
            
            popup_text += f"<b>Координаты:</b> {lat:.6f}, {lon:.6f}<br>"
            
            # Если есть ОКТМО
            if 'ОКТМО' in df.columns and not pd.isna(row['ОКТМО']):
                popup_text += f"<b>ОКТМО:</b> {row['ОКТМО']}<br>"
            
            popup_text += "</div>"
            
            # Определяем цвет по критичности
            marker_color = 'blue'
            if 'Крит  уровень ОЯ' in df.columns and not pd.isna(row['Крит  уровень ОЯ']):
                try:
                    crit_level = float(row['Крит  уровень ОЯ'])
                    if crit_level > 0:
                        marker_color = 'red'
                except Exception as e:
                    pass
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.7,
                weight=2
            ).add_to(m)
            
        except Exception as e:
            continue
    
    # Добавляем дополнительные плагины
    from folium.plugins import MiniMap, Fullscreen, MeasureControl
    MiniMap().add_to(m)
    Fullscreen().add_to(m)
    MeasureControl().add_to(m)
    
    # Слой контроля
    folium.LayerControl().add_to(m)
    
    # Отображаем карту
    from streamlit_folium import st_folium
    st_folium(m, width=800, height=600)

def show_yandex_map(gdf, df):
    """Показать карту Яндекс"""
    st.info("🟡 Яндекс.Карты - подробные карты России с актуальными данными")
    
    avg_lat = gdf.geometry.y.mean() if hasattr(gdf.geometry, 'y') else 62
    avg_lon = gdf.geometry.x.mean() if hasattr(gdf.geometry, 'x') else 129
    
    # Яндекс.Карты URL
    yandex_layers = {
        'Схема': 'https://vec01.maps.yandex.net/tiles?l=map&x={x}&y={y}&z={z}',
        'Спутник': 'https://vec01.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}',
        'Гибрид': 'https://vec01.maps.yandex.net/tiles?l=skl&x={x}&y={y}&z={z}'
    }
    
    selected_layer = st.selectbox("Выберите слой Яндекс.Карт", list(yandex_layers.keys()))
    
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=5,
        tiles=yandex_layers[selected_layer],
        attr='© Яндекс.Карты',
        control_scale=True,
        
    )
    
    # Добавляем маркеры
    for idx, row in gdf.iterrows():
        try:
            lat, lon = row.geometry.y, row.geometry.x
            
            # Иконка в зависимости от типа ГМС
            icon_color = 'blue'
            icon_type = 'info-sign'
            
            if 'ГМС' in df.columns:
                gms_name = str(row['ГМС']).lower()
                if any(word in gms_name for word in ['гидро', 'вод', 'уровень']):
                    icon_color = 'green'
                    icon_type = 'tint'
                elif any(word in gms_name for word in ['метео', 'погод', 'температур']):
                    icon_color = 'orange'
                    icon_type = 'cloud'
            
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{row.get('ГМС', f'Точка {idx}')}</b><br>{row.get('Река', '')}",
                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
            ).add_to(m)
            
        except Exception as e:
            continue
    
    from streamlit_folium import st_folium
    st_folium(m, width=800, height=600)

def show_dgis_map(gdf, df):
    """Показать карту 2ГИС"""
    st.info("🔵 2ГИС - подробная карта городов России с объектами инфраструктуры")
    
    avg_lat = gdf.geometry.y.mean() if hasattr(gdf.geometry, 'y') else 62
    avg_lon = gdf.geometry.x.mean() if hasattr(gdf.geometry, 'x') else 129
    
    # 2ГИС URL
    dgis_tiles = 'https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}'
    
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=5,
        tiles=dgis_tiles,
        attr='© 2ГИС',
        control_scale=True,
        
    )
    
    # Группируем точки по рекам для лучшей визуализации
    if 'Река' in df.columns:
        rivers = df['Река'].dropna().unique()
        river_colors = {}
        
        # Назначаем цвета рекам
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                 'lightblue', 'darkgreen', 'pink', 'gray']
        for i, river in enumerate(rivers[:len(colors)]):
            river_colors[river] = colors[i]
    
    # Добавляем маркеры
    for idx, row in gdf.iterrows():
        try:
            lat, lon = row.geometry.y, row.geometry.x
            
            # Определяем цвет по реке
            marker_color = 'blue'
            if 'Река' in df.columns and not pd.isna(row['Река']):
                marker_color = river_colors.get(row['Река'], 'blue')
            
            # Определяем размер по критичности
            marker_radius = 6
            if 'Крит  уровень ОЯ' in df.columns and not pd.isna(row['Крит  уровень ОЯ']):
                try:
                    crit_level = float(row['Крит  уровень ОЯ'])
                    if crit_level > 0:
                        marker_radius = 8 + min(crit_level / 100, 5)
                except Exception as e:
                    pass
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_radius,
                popup=f"<b>{row.get('ГМС', f'Точка {idx}')}</b><br>Река: {row.get('Река', '')}",
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.7,
                weight=2
            ).add_to(m)
            
        except Exception as e:
            continue
    
    from streamlit_folium import st_folium
    st_folium(m, width=800, height=600)

def show_map_comparison(gdf, df):
    """Показать сравнение разных карт"""
    st.info("🔍 Сравнение разных картографических сервисов")
    
    avg_lat = gdf.geometry.y.mean() if hasattr(gdf.geometry, 'y') else 62
    avg_lon = gdf.geometry.x.mean() if hasattr(gdf.geometry, 'x') else 129
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Яндекс.Карты")
        m_yandex = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=5,
            tiles='https://vec01.maps.yandex.net/tiles?l=map&x={x}&y={y}&z={z}',
            attr='© Яндекс.Карты',
            width=400,
            height=400,
            
        )
        
        # Добавляем несколько маркеров
        for idx, row in gdf.head(20).iterrows():
            try:
                lat, lon = row.geometry.y, row.geometry.x
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color='red',
                    fill=True
                ).add_to(m_yandex)
            except Exception as e:
                continue
        
        from streamlit_folium import folium_static
        folium_static(m_yandex)
    
    with col2:
        st.subheader("2ГИС")
        m_dgis = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=5,
            tiles='https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}',
            attr='© 2ГИС',
            width=400,
            height=400
        )
        
        # Добавляем несколько маркеров
        for idx, row in gdf.head(20).iterrows():
            try:
                lat, lon = row.geometry.y, row.geometry.x
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color='blue',
                    fill=True
                ).add_to(m_dgis)
            except Exception as e:
                continue
        
        folium_static(m_dgis)
    
    # Таблица сравнения
    st.subheader("📊 Сравнение картографических сервисов")
    
    comparison_data = {
        'Сервис': ['Яндекс.Карты', '2ГИС', 'Google Maps', 'OpenStreetMap'],
        'Покрытие России': ['Отличное', 'Хорошее (города)', 'Хорошее', 'Среднее'],
        'Детализация': ['Высокая', 'Очень высокая (города)', 'Высокая', 'Зависит от региона'],
        'Актуальность': ['Очень высокая', 'Высокая', 'Высокая', 'Переменная'],
        'Бесплатный доступ': ['Да', 'Да', 'Ограниченный', 'Да']
    }
    
    st.table(pd.DataFrame(comparison_data))

def show_geo_analysis():
    """Показать вкладку геоанализа"""
    st.header("🌍 Расширенная геоаналитика")
    
    # Проверяем наличие геоданных
    if 'current_gdf' not in st.session_state:
        st.warning("Сначала создайте карту на вкладке 'Карта'")
        return
    
    gdf = st.session_state['current_gdf']
    if gdf is None or len(gdf) == 0:
        st.warning("Нет геоданных для анализа")
        return
    
    df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    
    # Выбор инструментов анализа
    analysis_tool = st.selectbox(
        "Выберите инструмент анализа",
        ["Информация о точках", "Тепловая карта плотности", "Пространственная кластеризация", 
         "Поиск ближайших точек", "Статистика по регионам"]
    )
    
    if analysis_tool == "Информация о точках":
        st.subheader("📊 Информация о географических точках")
        
        # Показываем таблицу данных
        st.dataframe(df.head(20), use_container_width=True)
        
        # Основная статистика
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Всего точек", len(gdf))
            if hasattr(gdf, 'total_bounds'):
                bounds = gdf.total_bounds
                st.metric("Широта (мин-макс)", f"{bounds[1]:.4f} - {bounds[3]:.4f}")
        
        with col2:
            if 'Река' in df.columns:
                unique_rivers = df['Река'].nunique()
                st.metric("Уникальных рек", unique_rivers)
            
            if hasattr(gdf, 'total_bounds'):
                bounds = gdf.total_bounds
                st.metric("Долгота (мин-макс)", f"{bounds[0]:.4f} - {bounds[2]:.4f}")
    
    elif analysis_tool == "Тепловая карта плотности":
        st.subheader("🔥 Тепловая карта плотности")
        
        # Создаем карту для тепловой карты
        try:
            # Вычисляем средние координаты
            if hasattr(gdf.geometry, 'y'):
                avg_lat = gdf.geometry.y.mean()
                avg_lon = gdf.geometry.x.mean()
            else:
                avg_lat, avg_lon = 62, 129
            
            # Создаем базовую карту
            m = folium.Map(
                location=[avg_lat, avg_lon],
                zoom_start=5,
                tiles='CartoDB positron',
                control_scale=True,
                width='100%',
                height=500,
                
            )
            
            # Подготавливаем данные для тепловой карты
            heat_data = []
            for idx, row in gdf.iterrows():
                try:
                    if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                        lat, lon = row.geometry.y, row.geometry.x
                        if not pd.isna(lat) and not pd.isna(lon):
                            heat_data.append([lat, lon])
                except Exception as e:
                    continue
            
            # Добавляем тепловую карту если есть данные
            if heat_data:
                from folium.plugins import HeatMap
                
                HeatMap(
                    heat_data,
                    radius=15,
                    blur=20,
                    min_opacity=0.4,
                    max_zoom=10
                ).add_to(m)
                
                st.success(f"Тепловая карта создана по {len(heat_data)} точкам")
            else:
                st.warning("Нет данных для тепловой карты")
            
            # Отображаем карту
            from streamlit_folium import st_folium
            st_folium(m, width=800, height=500)
            
        except Exception as e:
            st.error(f"Ошибка создания тепловой карты: {e}")
    
    elif analysis_tool == "Пространственная кластеризация":
        st.subheader("🔢 Пространственная кластеризация")
        
        if len(gdf) < 2:
            st.warning("Нужно как минимум 2 точки для кластеризации")
            return
        
        # Настройки кластеризации
        n_clusters = st.slider(
            "Количество кластеров",
            min_value=2,
            max_value=min(10, len(gdf)),
            value=min(3, len(gdf)),
            step=1
        )
        
        if st.button("Выполнить кластеризацию", key="cluster_analysis_btn"):
            try:
                # Извлекаем координаты
                coords = []
                valid_indices = []
                
                for idx, row in gdf.iterrows():
                    try:
                        if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                            lon, lat = row.geometry.x, row.geometry.y
                            if not pd.isna(lat) and not pd.isna(lon):
                                coords.append([lon, lat])
                                valid_indices.append(idx)
                    except Exception as e:
                        continue
                
                if len(coords) < n_clusters:
                    st.warning(f"Нужно как минимум {n_clusters} точек для кластеризации, а есть {len(coords)}")
                    return
                
                # Выполняем кластеризацию
                from sklearn.cluster import KMeans
                import numpy as np
                
                coords_array = np.array(coords)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(coords_array)
                
                # Создаем DataFrame с результатами
                result_df = pd.DataFrame({
                    'Индекс': valid_indices,
                    'Широта': [coord[1] for coord in coords],
                    'Долгота': [coord[0] for coord in coords],
                    'Кластер': clusters
                })
                
                # Отображаем результаты
                st.subheader("📊 Результаты кластеризации")
                st.dataframe(result_df, use_container_width=True)
                
                # Создаем карту с цветами по кластерам
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 
                         'lightblue', 'darkgreen', 'pink', 'gray']
                
                m = folium.Map(
                    location=[result_df['Широта'].mean(), result_df['Долгота'].mean()],
                    zoom_start=5,
                    tiles='OpenStreetMap',
                    control_scale=True,
                    
                )
                
                for idx, row in result_df.iterrows():
                    color = colors[row['Кластер'] % len(colors)]
                    
                    folium.CircleMarker(
                        location=[row['Широта'], row['Долгота']],
                        radius=8,
                        popup=f"Кластер: {row['Кластер']}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        weight=1
                    ).add_to(m)
                
                # Добавляем центроиды кластеров
                for i, center in enumerate(kmeans.cluster_centers_):
                    folium.Marker(
                        location=[center[1], center[0]],
                        popup=f"Центроид кластера {i}",
                        icon=folium.Icon(color='black', icon='star')
                    ).add_to(m)
                
                # Отображаем карту
                from streamlit_folium import st_folium
                st_folium(m, width=800, height=500)
                
                # Статистика по кластерам
                st.subheader("📈 Статистика по кластерам")
                cluster_stats = pd.DataFrame({
                    'Кластер': range(n_clusters),
                    'Количество точек': np.bincount(clusters),
                    'Средняя широта': [result_df[result_df['Кластер'] == i]['Широта'].mean() for i in range(n_clusters)],
                    'Средняя долгота': [result_df[result_df['Кластер'] == i]['Долгота'].mean() for i in range(n_clusters)]
                })
                st.dataframe(cluster_stats, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка кластеризации: {e}")
    
    elif analysis_tool == "Поиск ближайших точек":
        st.subheader("🔍 Поиск ближайших точек")
        
        if len(gdf) < 2:
            st.warning("Нужно как минимум 2 точки для поиска")
            return
        
        # Выбор целевой точки
        point_options = list(range(min(50, len(gdf))))
        
        # Создаем список для отображения
        point_display = []
        for i in point_options:
            row = gdf.iloc[i]
            try:
                if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                    lat, lon = row.geometry.y, row.geometry.x
                    
                    # Берем название точки если есть
                    name = ""
                    if 'ГМС' in df.columns:
                        name = f" - {df.iloc[i]['ГМС']}"
                    elif 'Река' in df.columns:
                        name = f" - {df.iloc[i]['Река']}"
                    
                    point_display.append(f"Точка {i}{name} ({lat:.4f}, {lon:.4f})")
                else:
                    point_display.append(f"Точка {i}")
            except Exception as e:
                point_display.append(f"Точка {i}")
        
        target_idx = st.selectbox(
            "Выберите целевую точку",
            point_options,
            format_func=lambda x: point_display[x]
        )
        
        n_neighbors = st.slider(
            "Количество ближайших точек",
            min_value=1,
            max_value=min(10, len(gdf)-1),
            value=3,
            step=1
        )
        
        if st.button("Найти ближайшие точки", key="find_nearest_btn"):
            try:
                target_point = gdf.iloc[target_idx].geometry
                
                # Вычисляем расстояния до всех точек
                distances = []
                valid_points = []
                
                for i, row in gdf.iterrows():
                    if i == target_idx:
                        continue
                    
                    try:
                        if hasattr(row.geometry, 'distance'):
                            dist = target_point.distance(row.geometry)
                            if not pd.isna(dist):
                                distances.append(dist * 111.32)  # Конвертируем в км (1 градус ≈ 111.32 км)
                                valid_points.append(i)
                    except Exception as e:
                        continue
                
                if not distances:
                    st.warning("Не удалось вычислить расстояния")
                    return
                
                # Находим ближайшие точки - исправленная часть
                # Сортируем пары (расстояние, индекс)
                distance_pairs = list(zip(distances, valid_points))
                # Сортируем по расстоянию
                sorted_pairs = sorted(distance_pairs, key=lambda x: x[0])
                # Берем первые n_neighbors
                nearest_pairs = sorted_pairs[:n_neighbors]
                
                # Собираем результаты
                results = []
                for distance_km, point_idx in nearest_pairs:
                    # Получаем информацию о точке
                    point_info = {
                        'Индекс': point_idx,
                        'Расстояние (км)': distance_km
                    }
                    
                    # Добавляем доступные данные
                    if point_idx < len(df):
                        row_data = df.iloc[point_idx]
                        for col in ['ГМС', 'Река', 'Район', 'Наслег']:
                            if col in row_data:
                                point_info[col] = row_data[col]
                    
                    results.append(point_info)
                
                # Отображаем результаты
                st.subheader("📋 Ближайшие точки")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Создаем карту
                m = folium.Map(
                    location=[gdf.iloc[target_idx].geometry.y, gdf.iloc[target_idx].geometry.x],
                    zoom_start=7,
                    tiles='OpenStreetMap',
                    control_scale=True,
                    
                )
                
                # Добавляем целевую точку
                folium.CircleMarker(
                    location=[gdf.iloc[target_idx].geometry.y, gdf.iloc[target_idx].geometry.x],
                    radius=15,
                    popup=f"<b>Целевая точка</b><br>Индекс: {target_idx}",
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.8,
                    weight=3
                ).add_to(m)
                
                # Добавляем ближайшие точки
                for result in results:
                    point_idx = result['Индекс']
                    if point_idx < len(gdf):
                        point_geom = gdf.iloc[point_idx].geometry
                        
                        folium.CircleMarker(
                            location=[point_geom.y, point_geom.x],
                            radius=10,
                            popup=f"<b>Ближайшая точка</b><br>Индекс: {point_idx}<br>Расстояние: {result['Расстояние (км)']:.2f} км",
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.7,
                            weight=2
                        ).add_to(m)
                        
                        # Добавляем линию
                        folium.PolyLine(
                            locations=[[gdf.iloc[target_idx].geometry.y, gdf.iloc[target_idx].geometry.x],
                                     [point_geom.y, point_geom.x]],
                            color='blue',
                            weight=1,
                            opacity=0.5,
                            dash_array='5,5'
                        ).add_to(m)
                
                # Отображаем карту
                from streamlit_folium import st_folium
                st_folium(m, width=800, height=500)
                
            except Exception as e:
                st.error(f"Ошибка поиска ближайших точек: {e}")
    
    elif analysis_tool == "Статистика по регионам":
        st.subheader("📊 Статистика по регионам")
        
        # Проверяем наличие колонки с регионом
        region_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['район', 'наслег', 'октмо'])]
        
        if not region_cols:
            st.warning("В данных нет колонок с региональной информацией")
            return
        
        selected_region = st.selectbox("Выберите колонку для группировки", region_cols)
        
        if selected_region in df.columns:
            # Группируем данные
            region_stats = df.groupby(selected_region).agg({
                'Широта': ['count', 'mean', 'min', 'max'] if 'Широта' in df.columns else None,
                'Долгота': ['mean', 'min', 'max'] if 'Долгота' in df.columns else None
            }).reset_index()
            
            # Упрощаем имена колонок
            region_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in region_stats.columns]
            
            st.dataframe(region_stats, use_container_width=True)
            
            # Визуализация количества точек по регионам
            if 'Широта_count' in region_stats.columns:
                fig = px.bar(
                    region_stats,
                    x=selected_region,
                    y='Широта_count',
                    title=f"Количество точек по {selected_region}",
                    labels={'Широта_count': 'Количество точек', selected_region: selected_region}
                )
                st.plotly_chart(fig, use_container_width=True)
                
def show_predictions():
    """Показать вкладку предсказаний"""
    st.header("🔮 Предсказания временных рядов")
    
    # Получаем данные
    df = None
    
    if 'current_df' in st.session_state:
        df = st.session_state['current_df']
    elif 'special_coords_df' in st.session_state:
        df = st.session_state['special_coords_df']
    
    if df is None or df.empty:
        st.warning("Нет данных для анализа")
        return
    
    # Обнаруживаем колонки с датами
    date_columns = st.session_state.predictor.detect_date_columns(df)
    
    if not date_columns:
        st.warning("⚠️ Не найдены колонки с датами для временного анализа")
        st.info("Для предсказаний нужна хотя бы одна колонка с датами")
        return
    
    st.success(f"✅ Найдены колонки с датами: {', '.join(date_columns)}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Выбор колонки с датой
        date_column = st.selectbox(
            "Выберите колонку с датами:",
            date_columns,
            key="pred_date_col"
        )
        
        # Показываем информацию о датах
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce', format='mixed')
        df_copy = df_copy.dropna(subset=[date_column])
        
        if not df_copy.empty:
            date_range = df_copy[date_column]
            
            st.info(f"""
            **Диапазон дат:** {date_range.min().strftime('%Y-%m-%d')} - {date_range.max().strftime('%Y-%m-%d')}
            **Всего записей:** {len(df_copy)}
            **Уникальных дат:** {df_copy[date_column].nunique()}
            **Годы:** {df_copy[date_column].dt.year.min()} - {df_copy[date_column].dt.year.max()}
            """)
    
    # Выбор целевых переменных для прогнозирования
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_column in numeric_cols:
        numeric_cols.remove(date_column)
    
    if not numeric_cols:
        st.warning("Не найдены числовые колонки для прогнозирования")
        return
    
    st.subheader("🎯 Выбор параметров для прогнозирования")
    
    selected_targets = st.multiselect(
        "Выберите параметры для прогнозирования:",
        numeric_cols,
        default=numeric_cols[:2] if numeric_cols else [],
        key="pred_targets"
    )
    
    if not selected_targets:
        st.info("Выберите хотя бы один параметр для прогнозирования")
        return
    
    # Настройки прогнозирования
    with st.expander("⚙️ Настройки прогнозирования", expanded=True):
        col_set1, col_set2, col_set3 = st.columns(3)
        
        with col_set1:
            model_type = st.selectbox(
                "Модель прогнозирования:",
                ["xgboost", "linear"],
                format_func=lambda x: {
                    "xgboost": "XGBoost (Градиентный бустинг)",
                    "linear": "Линейная базовая регрессия"
                }[x],
                key="pred_model"
            )
            
            forecast_type = st.selectbox(
                "Тип прогноза:",
                ["На следующий год", "На конкретную дату"],
                key="forecast_type"
            )
        
        with col_set2:
            # Прогноз для конкретной даты
            if forecast_type == "На конкретную дату":
                specific_date = st.date_input(
                    "Прогноз на конкретную дату:",
                    value=datetime.now() + timedelta(days=30)
                )
            
            # Анализ сезонности
            analyze_seasonality = st.checkbox("Анализ сезонности", value=True)
        
        with col_set3:
            # Дополнительные опции
            calculate_metrics = st.checkbox("Рассчитать метрики качества", value=True)
            show_details = st.checkbox("Показать детальный отчет", value=True)
            save_predictions = st.checkbox("Сохранить прогнозы", value=True)
    
    # Кнопка запуска прогнозирования
    if st.button("🚀 Начать прогнозирование", type="primary", key="start_prediction"):
        progress_bar = st.progress(0)
        
        results = {}
        all_reports = {}
        
        for i, target in enumerate(selected_targets):
            progress = (i) / len(selected_targets)
            progress_bar.progress(progress)
            
            with st.spinner(f"Обучаю модель для {target}..."):
                try:
                    # Обучаем модель
                    score = st.session_state.predictor.train_model(
                        df, target, date_column, model_type
                    )
                    
                    st.success(f"✅ Модель для '{target}' обучена (R² = {score:.3f})")
                    
                    # В зависимости от типа прогноза
                    if forecast_type == "На следующий год":
                        # Прогноз на следующий год
                        st.subheader(f"📅 Прогноз для '{target}' на следующий год")
                        
                        # Получаем детальный отчет
                        report = None
                        if hasattr(st.session_state.predictor, 'create_detailed_forecast_report'):
                            report = st.session_state.predictor.create_detailed_forecast_report(
                                df, target, date_column
                            )
                        
                        if report:
                            # Показываем основную сводку
                            if 'summary' in report:
                                st.info(report['summary'])
                            
                            # Показываем ключевые даты
                            if 'key_dates' in report and report['key_dates']:
                                st.subheader("🗓️ Ключевые даты")
                                for key_date in report['key_dates'][:3]:
                                    st.write(f"**{key_date.get('description', 'Дата')}:** {key_date.get('value', 'N/A')} ({key_date.get('date', 'N/A')})")
                            
                            # Показываем рекомендации
                            if 'recommendations' in report and report['recommendations']:
                                st.subheader("💡 Рекомендации")
                                for rec in report['recommendations']:
                                    st.write(f"• {rec}")
                            
                            all_reports[target] = report
                        
                        # Создаем визуализацию
                        fig = None
                        if hasattr(st.session_state.predictor, 'create_visual_forecast'):
                            fig = st.session_state.predictor.create_visual_forecast(
                                df, target, date_column
                            )
                        elif hasattr(st.session_state.predictor, 'create_forecast_plot'):
                            fig = st.session_state.predictor.create_forecast_plot(
                                df, target, date_column
                            )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=unique_key())
                    
                    elif forecast_type == "На конкретную дату":
                        # Прогноз на конкретную дату
                        st.subheader(f"🎯 Прогноз для '{target}' на {specific_date.strftime('%d.%m.%Y')}")
                        
                        try:
                            result = None
                            # Пробуем исправленную версию
                            if hasattr(st.session_state.predictor, 'predict_for_specific_date_fixed'):
                                result = st.session_state.predictor.predict_for_specific_date_fixed(
                                    df, target, date_column, specific_date.strftime('%d.%m.%Y')
                                )
                            elif hasattr(st.session_state.predictor, 'predict_for_specific_date'):
                                result = st.session_state.predictor.predict_for_specific_date(
                                    df, target, date_column, specific_date.strftime('%d.%m.%Y')
                                )
                            
                            if result:
                                col_pred1, col_pred2, col_pred3 = st.columns(3)
                                
                                with col_pred1:
                                    st.metric(
                                        "Ожидаемое значение",
                                        f"{result.get('value', 0):.1f}",
                                        delta=f"±{result.get('confidence', 'высокая')}"
                                    )
                                
                                with col_pred2:
                                    if 'date' in result:
                                        st.metric(
                                            "Ближайшая дата",
                                            result['date']
                                        )
                                
                                with col_pred3:
                                    if 'days_difference' in result:
                                        st.metric(
                                            "Разница в днях",
                                            f"{result['days_difference']} дн."
                                        )
                                
                                # Дополнительная информация
                                if 'exact_match' in result:
                                    st.info(f"**Точное совпадение:** {'Да' if result['exact_match'] else 'Нет'}")
                                if 'based_on_dates' in result:
                                    st.info(f"**Основано на данных от:** {', '.join(result['based_on_dates'])}")
                        
                        except Exception as e:
                            st.error(f"Ошибка прогноза на конкретную дату: {str(e)}")
                    
                    # Анализ сезонности
                    if analyze_seasonality:
                        st.subheader("🌦️ Анализ сезонных паттернов")
                        season_fig = None
                        if hasattr(st.session_state.predictor, 'analyze_seasonal_patterns'):
                            season_fig = st.session_state.predictor.analyze_seasonal_patterns(
                                df, target, date_column
                            )
                        
                        if season_fig:
                            st.plotly_chart(season_fig, use_container_width=True)
                        else:
                            st.warning("График сезонности недоступен")
                    
                    # Метрики качества
                    if calculate_metrics:
                        metrics = {}
                        if hasattr(st.session_state.predictor, 'get_forecast_metrics'):
                            metrics = st.session_state.predictor.get_forecast_metrics(
                                df, target, date_column
                            )
                        
                        if metrics:
                            st.subheader("📊 Метрики качества прогноза")
                            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                            
                            if 'MAE_mean' in metrics:
                                col_met1.metric("MAE", f"{metrics['MAE_mean']:.3f}")
                            elif 'MAE' in metrics:
                                col_met1.metric("MAE", f"{metrics['MAE']:.3f}")
                            
                            if 'RMSE_mean' in metrics:
                                col_met2.metric("RMSE", f"{metrics['RMSE_mean']:.3f}")
                            elif 'RMSE' in metrics:
                                col_met2.metric("RMSE", f"{metrics['RMSE']:.3f}")
                            
                            if 'R2_mean' in metrics:
                                col_met3.metric("R²", f"{metrics['R2_mean']:.3f}")
                            elif 'R2' in metrics:
                                col_met3.metric("R²", f"{metrics['R2']:.3f}")
                            
                            if 'cv_folds' in metrics:
                                col_met4.metric("Кросс-валидация", f"{metrics['cv_folds']} фолдов")
                            elif 'MAPE' in metrics:
                                col_met4.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                    
                    # Важность признаков
                    if hasattr(st.session_state.predictor, 'feature_importance'):
                        imp_df = st.session_state.predictor.feature_importance.get(target)
                        if imp_df is not None and not imp_df.empty:
                            st.subheader("🔍 Важность признаков")
                            fig_imp = px.bar(
                                imp_df.head(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title=f"Топ-10 важных признаков для '{target}'"
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Сохраняем результаты
                    results[target] = {
                        'score': score,
                        'metrics': metrics if calculate_metrics else {}
                    }
                    
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании '{target}': {str(e)}")
        
        progress_bar.progress(1.0)
        
        # Комплексный анализ для речных данных
        st.markdown("---")
        
        # Сохранение результатов
        if save_predictions and (results or all_reports):
            st.markdown("---")
            st.subheader("💾 Сохранение результатов")
            
            datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Сохраняем модели
            models_saved = 0
            for target in results.keys():
                try:
                    if hasattr(st.session_state.predictor, 'save_model'):
                        model_filename = f"reports/models/model_{target}_{unique_key}.pkl"
                        Path(model_filename).parent.mkdir(exist_ok=True, parents=True)
                        if st.session_state.predictor.save_model(target, model_filename):
                            models_saved += 1
                except Exception as e:
                    continue
            
            if models_saved > 0:
                st.success(f"✅ Сохранено {models_saved} моделей")
    
    # Пример использования для обучения
    st.markdown("---")
    with st.expander("📚 Как использовать инструмент прогнозирования"):
        st.markdown("""
        ### 📅 Типы прогнозов:
        
        1. **На следующий год** - полный прогноз на все 365 дней следующего года
        2. **На конкретную дату** - точный прогноз на выбранную дату
        
        ### 🎯 Конкретные прогнозы показывают:
        - **Точные даты** максимумов и минимумов
        - **Численные значения** с доверительными интервалами
        - **Рекомендации** на основе анализа
        - **Визуализации** трендов и сезонности
        
        ### 📊 Пример вывода:
        ```
        Прогноз температуры на 2025 год:
        - Максимальная температура: 25.3°C (15.07.2025)
        - Минимальная температура: -12.4°C (15.01.2025)
        - Среднегодовая: 8.7°C
        
        Рекомендации:
        • Ожидается жаркое лето, будьте готовы к высокой температуре
        • Холодная зима, подготовьтесь к морозам
        ```
        
        ### 🌊 Особенности для речных данных:
        - Автоматическое определение параметров (температура, осадки, уровень воды)
        - Анализ рисков (паводки, засухи)
        - Многолетние тренды (1960-2025 гг.)
        """)

def show_dashboard():
    """Показать вкладку дашборда"""
    st.header("📈 Интерактивный дашборд")
    
    # Получаем данные
    has_coords = st.session_state.get('has_coords', False)
    df = None
    
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        if not gdf.empty:
            df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    elif 'current_df' in st.session_state:
        df = st.session_state['current_df']
    
    if df is None or df.empty:
        st.warning("Нет данных для анализа")
        return
    
    # Автоматическая визуализация
    st.subheader("🚀 Автоматическая визуализация")
    
    target_column = st.selectbox(
        "Выберите целевую переменную (опционально)",
        ["Нет"] + df.columns.tolist(),
        key="dashboard_target"
    )
    if target_column == "Нет":
        target_column = None
    
    if st.button("🤖 Создать автоматический дашборд", type="primary", key="auto_dashboard_btn"):
        with st.spinner("Создаю дашборд..."):
            visualizations = st.session_state.visualizer.auto_visualize(
                df, 
                target_column=target_column,
                max_columns=8
            )
            
            if visualizations:
                st.success(f"Создано {len(visualizations)} визуализаций")
                
                # Настройки подсказок для всех графиков
                hover_settings = {
                    'bgcolor': '#FFFFFF',
                    'font_size': 12,
                    'font_color': '#000000',
                    'bordercolor': '#CCCCCC'
                }
                
                # Применяем настройки подсказок ко всем графикам
                for viz in visualizations:
                    viz.update_layout(hoverlabel=hover_settings)
                
                # Отображаем визуализации в сетке
                cols_per_row = 2
                for i in range(0, len(visualizations), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(visualizations):
                            with cols[j]:
                                st.plotly_chart(visualizations[i + j], use_container_width=True)
            else:
                st.warning("Не удалось создать автоматические визуализации")
    
    st.markdown("---")
    st.subheader("🎯 Целевой анализ")
    
    if target_column and target_column in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Анализ зависимости от целевой переменной
            if pd.api.types.is_numeric_dtype(df[target_column]):
                st.markdown(f"### 📊 Корреляции с '{target_column}'")
                
                # Находим наибольшие корреляции
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                correlations = df[numeric_cols].corr()[target_column].drop(target_column)
                top_correlations = correlations.abs().sort_values(ascending=False).head(5)
                
                for col in top_correlations.index:
                    corr_val = correlations[col]
                    fig = st.session_state.visualizer.create_scatter_plot(
                        df, col, target_column,
                        title=f"{col} vs {target_column} (корреляция: {corr_val:.3f})",
                        trendline='ols'
                    )
                    # Применяем настройки подсказок
                    fig.update_layout(
                        hoverlabel=dict(
                            bgcolor='#FFFFFF',
                            font_size=12,
                            font_color='#000000'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True, key=unique_key())
            
            else:
                # Для категориальной целевой переменной
                st.markdown(f"### 📊 Распределение по '{target_column}'")
                
                # Круговая диаграмма
                if df[target_column].nunique() <= 10:
                    fig = st.session_state.visualizer.create_pie_chart(
                        df, target_column,
                        title=f"Распределение по {target_column}",
                        hole=0.4
                    )
                    fig.update_layout(
                        hoverlabel=dict(
                            bgcolor='#FFFFFF',
                            font_size=12,
                            font_color='#000000'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plots для числовых переменных
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols[:2]:
                        fig = st.session_state.visualizer.create_box_plot(
                            df, target_column, col,
                            title=f"{col} по {target_column}"
                        )
                        fig.update_layout(
                            hoverlabel=dict(
                                bgcolor='#FFFFFF',
                                font_size=12,
                                font_color='#000000'
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Метрики
            st.markdown("### 📈 Ключевые метрики")
            
            if pd.api.types.is_numeric_dtype(df[target_column]):
                metrics_data = {
                    "Среднее": df[target_column].mean(),
                    "Медиана": df[target_column].median(),
                    "Стандартное отклонение": df[target_column].std(),
                    "Минимум": df[target_column].min(),
                    "Максимум": df[target_column].max(),
                    "Квантиль 25%": df[target_column].quantile(0.25),
                    "Квантиль 75%": df[target_column].quantile(0.75)
                }
                
                for metric_name, metric_value in metrics_data.items():
                    if pd.notna(metric_value):
                        st.metric(metric_name, f"{metric_value:.2f}")
                
                # Гистограмма распределения
                fig = st.session_state.visualizer.create_histogram(
                    df, target_column,
                    title=f"Распределение {target_column}",
                    histnorm='probability density'
                )
                fig.update_layout(
                    hoverlabel=dict(
                        bgcolor='#FFFFFF',
                        font_size=12,
                        font_color='#000000'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Для категориальных данных
                value_counts = df[target_column].value_counts()
                
                st.metric("Уникальных значений", len(value_counts))
                st.metric("Самое частое", value_counts.index[0])
                st.metric("Частота самого частого", value_counts.iloc[0])
                
                # Барчарт распределения
                top_values = value_counts.head(10)
                fig_df = pd.DataFrame({
                    'Категория': top_values.index,
                    'Количество': top_values.values
                })
                
                fig = st.session_state.visualizer.create_bar_chart(
                    fig_df, 'Категория', 'Количество',
                    title=f"Топ-10 значений {target_column}",
                    orientation='h',
                    color_palette='bold'
                )
                fig.update_layout(
                    hoverlabel=dict(
                        bgcolor='#FFFFFF',
                        font_size=12,
                        font_color='#000000'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Матрица корреляций
    st.markdown("---")
    st.subheader("🌡️ Корреляционная матрица")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        fig = st.session_state.visualizer.create_correlation_matrix(
            df[numeric_cols],
            title="Тепловая карта корреляций",
            height=600,
            color_scale='rdbu'
        )
        
        if fig:
            # Применяем настройки подсказок
            fig.update_layout(
                hoverlabel=dict(
                    bgcolor='#FFFFFF',
                    font_size=12,
                    font_color='#000000'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Самые сильные корреляции
            st.subheader("🔗 Самые сильные корреляции")
            
            # Используем анализатор для получения корреляций
            corr_df = st.session_state.analyzer.get_top_correlations(df[numeric_cols], 10)
            if not corr_df.empty:
                st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("Для корреляционного анализа нужно как минимум 2 числовых колонок")
    
    # Экспорт дашборда
    st.markdown("---")
    if st.button("💾 Сохранить дашборд как HTML", type="primary", key="save_dash_btn"):
        datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_path = Path("reports") / "dashboards" / f"дашборд_{st.session_state.get('current_file', 'data')}_{unique_key }.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Дашборд анализа данных - {st.session_state.get('current_file', 'data')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 10px; }}
                .metric {{ background-color: #e7f3fe; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .hoverlayer {{ background-color: rgba(255, 255, 255, 0.95) !important; }}
                .hovertext {{ background-color: white !important; color: #333 !important; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Дашборд анализа данных</h1>
                <p><strong>Файл:</strong> {st.session_state.get('current_file', 'data')}</p>
                <p><strong>Дата создания:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Размер данных:</strong> {len(df)} строк × {len(df.columns)} колонок</p>
            </div>
        </body>
        </html>
        """
        
        dashboard_path.parent.mkdir(exist_ok=True, parents=True)
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        st.success(f"✅ Дашборд сохранен: {dashboard_path}")

def show_special_coordinates():
    """Показать вкладку специальных координат"""
    st.header("📍 Специальные координаты")
    
    if 'special_coords_df' not in st.session_state:
        st.warning("Нет данных о специальных координатах")
        return
    
    df = st.session_state['special_coords_df']
    gdf = st.session_state.get('special_coords_gdf', None)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Данные координат")
        st.dataframe(df, use_container_width=True)
        
        # Статистика
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("📈 Статистика")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Широта мин", f"{df['latitude'].min():.6f}")
                st.metric("Широта макс", f"{df['latitude'].max():.6f}")
            
            with col_stat2:
                st.metric("Долгота мин", f"{df['longitude'].min():.6f}")
                st.metric("Долгота макс", f"{df['longitude'].max():.6f}")
            
            with col_stat3:
                st.metric("Всего точек", len(df))
                st.metric("Уникальных точек", df[['latitude', 'longitude']].drop_duplicates().shape[0])
    
    with col2:
        st.subheader("🛠️ Инструменты")
        
        # Опция для создания карты
        if st.button("🗺️ Создать карту", type="primary", key="create_spec_map_btn"):
            if gdf is None:
                # Создаем GeoDataFrame
                try:
                    from shapely.geometry import Point
                    import geopandas as gpd
                    
                    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                    st.session_state['special_coords_gdf'] = gdf
                    st.success("Геоданные созданы!")
                except Exception as e:
                    st.error(f"Ошибка создания геоданных: {e}")
                    return
            
            # Создаем карту
            m = st.session_state.visualizer.create_base_map(
                center_lat=df['latitude'].mean(),
                center_lon=df['longitude'].mean(),
                zoom_start=4,
                height=500,
                width=700
            )
            
            # Добавляем маркеры
            m = st.session_state.visualizer.add_markers_to_map(
                m, gdf, df,
                cluster=True,
                radius=8,
                opacity=0.7,
                color_palette='set1'
            )
            
            st.session_state['special_coords_map'] = m
            st.success("Карта создана!")
        
        # Опция для тепловой карты
        if st.button("🔥 Создать тепловую карту", key="create_spec_heat_btn"):
            if gdf is not None:
                heatmap = st.session_state.visualizer.create_heatmap(
                    gdf,
                    radius=15,
                    blur=20,
                    min_opacity=0.4,
                    height=500,
                    width=700
                )
                st.session_state['special_coords_heatmap'] = heatmap
                st.success("Тепловая карта создана!")
        
        # Экспорт данных
        st.markdown("---")
        if st.button("💾 Экспорт в CSV", key="export_spec_csv_btn"):
            csv_path = Path("reports") / "coords" / f"координаты_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            st.success(f"Данные сохранены: {csv_path}")
        
        if st.button("🌍 Экспорт в GeoJSON", key="export_spec_geojson_btn"):
            if gdf is not None:
                geojson_path = Path("reports") / "coords" / f"координаты_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
                geojson_path.parent.mkdir(exist_ok=True, parents=True)
                gdf.to_file(geojson_path, driver='GeoJSON')
                st.success(f"GeoJSON сохранен: {geojson_path}")
    
    # Отображение карт
    if 'special_coords_map' in st.session_state:
        st.subheader("🗺️ Карта координат")
        from streamlit_folium import st_folium
        st_folium(st.session_state['special_coords_map'], width=800, height=500)
    
    if 'special_coords_heatmap' in st.session_state:
        st.subheader("🔥 Тепловая карта")
        from streamlit_folium import st_folium
        st_folium(st.session_state['special_coords_heatmap'], width=800, height=500)
    
    # Анализ кластеризации
    st.markdown("---")
    st.subheader("🔍 Анализ кластеризации")
    
    if gdf is not None and len(gdf) > 1:
        n_clusters = st.slider("Количество кластеров", 2, min(10, len(gdf)), 3)
        
        if st.button("🔢 Выполнить кластеризацию", key="cluster_spec_btn"):
            from sklearn.cluster import KMeans
            
            coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(coords)
            
            # Добавляем кластеры в данные
            df_clustered = df.copy()
            df_clustered['cluster'] = clusters
            
            # Визуализация
            st.subheader(f"Кластеризация ({n_clusters} кластера)")
            
            # Цвета для кластеров
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
            
            # Создаем карту с цветами по кластерам
            m_clusters = st.session_state.visualizer.create_base_map(
                center_lat=df['latitude'].mean(),
                center_lon=df['longitude'].mean(),
                zoom_start=4,
                height=500,
                width=700
            )
            
            # Добавляем маркеры с цветами по кластерам
            for idx, row in df_clustered.iterrows():
                color = colors[row['cluster'] % len(colors)]
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=f"Кластер: {row['cluster']}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1
                ).add_to(m_clusters)
            
            # Отображаем карту
            from streamlit_folium import st_folium
            st_folium(m_clusters, width=800, height=500)
            
            # Статистика по кластерам
            st.subheader("📊 Статистика по кластерам")
            cluster_stats = pd.DataFrame({
                'Кластер': range(n_clusters),
                'Количество точек': np.bincount(clusters),
                'Средняя широта': [df_clustered[df_clustered['cluster'] == i]['latitude'].mean() for i in range(n_clusters)],
                'Средняя долгота': [df_clustered[df_clustered['cluster'] == i]['longitude'].mean() for i in range(n_clusters)]
            })
            st.dataframe(cluster_stats, use_container_width=True)

def render_footer():
    """Рендер футера"""
    st.markdown("---")
    st.caption(" Анализатор данных с геоаналитикой | Создано с помощью Streamlit, GeoPandas, Plotly")