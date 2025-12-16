import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import folium
from sklearn.cluster import KMeans
from .analyzer import GeoDataAnalyzer
from .visualization import Visualizer

def setup_app():
    """Настройка приложения"""
    st.set_page_config(
        page_title="🌍 Анализатор данных с геоаналитикой",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Инициализация компонентов в session_state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GeoDataAnalyzer()
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
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
    
    st.title("🌍 Анализатор данных с расширенной геоаналитикой")
    st.markdown("---")

def render_sidebar():
    """Рендер сайдбара"""
    # Убедимся, что анализатор инициализирован
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GeoDataAnalyzer()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
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
    if 'current_gdf' not in st.session_state and 'current_df' not in st.session_state:
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
    filename = st.session_state.get('current_file', 'Файл')
    has_coords = st.session_state.get('has_coords', False)
    has_special_coords = st.session_state.get('has_special_coords', False)
    
    # Определяем тип данных
    if has_coords and 'current_gdf' in st.session_state:
        data_type = "🌍 Геоданные"
    elif has_special_coords and 'special_coords_df' in st.session_state:
        data_type = "📍 Специальные координаты"
    else:
        data_type = "📊 Обычные данные"
    
    # Создаем вкладки
    tabs = ["📋 Обзор данных", "📊 Визуализация"]
    
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
    
    # Вкладка 2: Визуализация
    with tab_objects[1]:
        show_visualization()
    
    if has_special_coords and len(tabs) > 2:
        special_coords_idx = 2  # После визуализации
        with tab_objects[special_coords_idx]:
            show_special_coordinates()
            
    # Вкладка 3: Карта (только для геоданных)
    if has_coords and len(tabs) > 2:
        with tab_objects[2]:
            show_map_view()
    
    # Вкладка 4: Геоаналитика (только для геоданных)
    if has_coords and len(tabs) > 3:
        with tab_objects[3]:
            show_geo_analysis()
    
    # Вкладка 5: Дашборд
    dashboard_tab_idx = 4 if has_coords else 2
    with tab_objects[dashboard_tab_idx]:
        show_dashboard()

def load_file_from_folder(selected_file):
    """Загрузить файл из папки"""
    file_path = Path(st.session_state.analyzer.data_folder) / selected_file
    df = st.session_state.analyzer.load_file(file_path)
    
    if df is not None:
        process_loaded_data(df, selected_file)

def load_uploaded_file(uploaded_file):
    """Загрузить загруженный файл"""
    try:
        temp_path = Path("temp") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        df = st.session_state.analyzer.load_file(temp_path)
        
        if df is not None:
            process_loaded_data(df, uploaded_file.name)
            temp_path.unlink()
            
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")

def process_loaded_data(df, filename):
    """Обработать загруженные данные"""
    # Проверяем стандартные координаты
    lat_cols, lon_cols = st.session_state.analyzer.detect_coordinates(df)
    
    # Проверяем специальные координаты в формате N/E
    special_lat_cols = [col for col in df.columns if any(word in str(col).lower() for word in ['широт', 'latitude', 'lat', 'геопозиция'])]
    special_lon_cols = [col for col in df.columns if any(word in str(col).lower() for word in ['долгот', 'longitude', 'lon', 'lng', 'геопозиция'])]
    
    has_special_format = False
    if len(special_lat_cols) >= 1 and len(special_lon_cols) >= 1:
        # Проверяем, есть ли данные в формате N/E
        sample_value = str(df[special_lat_cols[0]].iloc[0]) if len(df) > 0 else ""
        if sample_value and (sample_value.startswith('N') or sample_value.startswith('S')):
            has_special_format = True
    
    if lat_cols and lon_cols:
        # Стандартные координаты
        st.info(f"📌 Найдены координаты: Широта='{lat_cols[0]}', Долгота='{lon_cols[0]}'")
        gdf = st.session_state.analyzer.create_geodataframe(df, lat_cols[0], lon_cols[0])
        st.session_state['current_gdf'] = gdf
        st.session_state['has_coords'] = True
        st.session_state['lat_col'] = lat_cols[0]
        st.session_state['lon_col'] = lon_cols[0]
    
    elif has_special_format:
        # Специальные координаты в формате N/E
        lat_col = special_lat_cols[0]
        lon_col = special_lon_cols[0]
        
        st.info(f"📌 Найдены специальные координаты: '{lat_col}', '{lon_col}'")
        st.info("Формат: N/E (например, N67.45682409, E153.71599186)")
        
        # Преобразуем координаты
        df_converted = df.copy()
        df_converted['latitude'] = df_converted[lat_col].apply(
            lambda x: st.session_state.analyzer.convert_dms_to_decimal(x)
        )
        df_converted['longitude'] = df_converted[lon_col].apply(
            lambda x: st.session_state.analyzer.convert_dms_to_decimal(x)
        )
        
        # Удаляем строки с None
        df_converted = df_converted.dropna(subset=['latitude', 'longitude'])
        
        if len(df_converted) > 0:
            st.success(f"✅ Преобразовано {len(df_converted)} координат")
            
            # Сохраняем обработанные данные
            st.session_state['special_coords_df'] = df_converted
            st.session_state['has_special_coords'] = True
            st.session_state['has_coords'] = False
            
            # Показываем пример преобразованных координат
            st.subheader("Пример преобразованных координат:")
            st.dataframe(df_converted[['latitude', 'longitude'] + list(df.columns)].head(5))
        else:
            st.warning("Не удалось преобразовать координаты")
            st.session_state['current_df'] = df
            st.session_state['has_coords'] = False
    
    elif hasattr(df, 'geometry') or 'geometry' in df.columns:
        st.session_state['current_gdf'] = df
        st.session_state['has_coords'] = True
    
    else:
        possible_lat = [col for col in df.columns if any(word in str(col).lower() for word in ['широт', 'latitude', 'lat'])]
        possible_lon = [col for col in df.columns if any(word in str(col).lower() for word in ['долгот', 'longitude', 'lon', 'lng'])]
        
        if possible_lat or possible_lon:
            st.warning(f"⚠️ Возможные координатные столбцы: {possible_lat + possible_lon}")
            st.info("📌 Для ручного указания используйте инструменты геоаналитики")
        
        st.session_state['current_df'] = df
        st.session_state['has_coords'] = False
    
    st.session_state['current_file'] = filename
    st.success(f"Файл {filename} загружен успешно!")
    
# Страницы вкладок
def show_data_overview():
    """Показать вкладку обзора данных"""
    st.header("📋 Обзор данных")
    
    # Получаем данные из сессии
    has_coords = st.session_state.get('has_coords', False)
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    else:
        df = st.session_state.get('current_df')
        gdf = None
    
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
    if gdf is not None:
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

def show_visualization():
    """Показать вкладку визуализации"""
    st.header("📊 Интерактивная визуализация")
    
    # Получаем данные
    has_coords = st.session_state.get('has_coords', False)
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    else:
        df = st.session_state.get('current_df')
    
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
    
    if st.button("🔄 Создать график", type="primary", key="create_viz"):
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
            
            if fig:
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
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.visualizer.default_config)
                
                # Опции сохранения
                with st.expander("💾 Опции сохранения"):
                    col_save1, col_save2, col_save3 = st.columns(3)
                    
                    with col_save1:
                        if st.button("Сохранить как HTML"):
                            saved_files = st.session_state.visualizer.save_plot(
                                fig, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                formats=['html']
                            )
                            if saved_files:
                                st.success(f"Сохранено: {saved_files[0]}")
                    
                    with col_save2:
                        if st.button("Сохранить как PNG"):
                            saved_files = st.session_state.visualizer.save_plot(
                                fig, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                formats=['png']
                            )
                            if saved_files:
                                st.success(f"Сохранено: {saved_files[0]}")
                    
                    with col_save3:
                        if st.button("Сохранить все форматы"):
                            saved_files = st.session_state.visualizer.save_plot(
                                fig, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                formats=['html', 'png', 'json']
                            )
                            if saved_files:
                                st.success(f"Сохранено {len(saved_files)} файлов")
                
        except Exception as e:
            st.error(f"Ошибка создания графика: {e}")

def show_map_view():
    """Показать вкладку карты"""
    st.header("🗺️ Интерактивная карта")
    
    gdf = st.session_state.get('current_gdf')
    df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Настройки карты
        map_type = st.selectbox(
            "Тип карты",
            ["OpenStreetMap", "Stamen Terrain", "Stamen Toner", "CartoDB positron"],
            key="map_type"
        )
        
        # Выбор колонки для цвета
        color_options = ["Нет"] + df.columns.tolist()
        color_by = st.selectbox("Цвет по колонке", color_options, key="map_color")
        
        # Выбор колонки для размера
        size_options = ["Нет"] + df.select_dtypes(include=[np.number]).columns.tolist()
        size_by = st.selectbox("Размер по колонке", size_options, key="map_size")
        if size_by == "Нет":
            size_by = None
        
        # Выбор колонок для подсказки
        popup_options = ["Все"] + df.columns.tolist()
        popup_cols = st.multiselect("Колонки в подсказке", popup_options, default=popup_options[:5])
        
        # Кластеризация маркеров
        use_clustering = st.checkbox("Использовать кластеризацию маркеров", value=True)
    
    with col2:
        # Настройки отображения
        marker_radius = st.slider("Радиус маркера", 3, 20, 8)
        marker_opacity = st.slider("Прозрачность маркера", 0.1, 1.0, 0.7, 0.1)
        
        # Цветовая палитра
        palettes = st.session_state.visualizer.get_available_color_palettes()
        color_palette = st.selectbox("Палитра для цвета", palettes[:10], key="map_palette")
    
    with col3:
        # Выбор колонки для значения
        value_options = ["Нет"] + df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = st.selectbox("Колонка для тепловой карты", value_options, key="map_value")
        if value_col == "Нет":
            value_col = None
        
        # Создание карты
        if gdf is not None and st.button("🗺️ Создать/Обновить карту", type="primary"):
            # Создаем базовую карту
            m = st.session_state.visualizer.create_base_map(
                gdf=gdf,
                tiles=map_type,
                height=600,
                width=800
            )
            
            # Добавляем маркеры
            m = st.session_state.visualizer.add_markers_to_map(
                m, gdf, df,
                color_by=color_by if color_by != "Нет" else None,
                size_by=size_by,
                popup_cols=popup_cols if popup_cols != ["Все"] else df.columns.tolist()[:5],
                cluster=use_clustering,
                radius=marker_radius,
                opacity=marker_opacity,
                color_palette=color_palette
            )
            
            # Сохраняем карту в сессии для отображения
            st.session_state['current_map'] = m
            
            # Тепловая карта
            if value_col:
                heatmap = st.session_state.visualizer.create_heatmap(
                    gdf, value_col,
                    radius=20,
                    blur=15,
                    min_opacity=0.4
                )
                st.session_state['current_heatmap'] = heatmap
    
    # Отображение карты
    if 'current_map' in st.session_state:
        from streamlit_folium import st_folium
        
        # Выбор типа отображения
        display_type = st.radio(
            "Тип отображения",
            ["Маркеры", "Тепловая карта", "Оба"],
            horizontal=True
        )
        
        if display_type == "Маркеры":
            st_folium(st.session_state['current_map'], width=800, height=600)
        elif display_type == "Тепловая карта" and 'current_heatmap' in st.session_state:
            st_folium(st.session_state['current_heatmap'], width=800, height=600)
        elif display_type == "Оба" and 'current_heatmap' in st.session_state:
            # Комбинированное отображение
            st.info("Комбинированное отображение (сначала маркеры, затем тепловая карта)")
            st_folium(st.session_state['current_map'], width=800, height=600)
            st_folium(st.session_state['current_heatmap'], width=800, height=600)
        
        # Опции экспорта
        with st.expander("📤 Опции экспорта"):
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if st.button("💾 Сохранить карту как HTML"):
                    map_path = Path("reports") / "visualizations" / f"карта_{st.session_state.get('current_file', 'data')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    map_path.parent.mkdir(exist_ok=True, parents=True)
                    st.session_state['current_map'].save(str(map_path))
                    st.success(f"Карта сохранена: {map_path}")
                    
                    # Предложение скачать
                    with open(map_path, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    
                    st.download_button(
                        label="📥 Скачать карту",
                        data=map_html,
                        file_name=f"карта_{st.session_state.get('current_file', 'data')}.html",
                        mime="text/html"
                    )
            
            with col_exp2:
                if st.button("🌍 Экспорт в GeoJSON"):
                    geojson_path = st.session_state.analyzer.export_to_geojson(
                        gdf, 
                        f"{st.session_state.get('current_file', 'data')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
                    )
                    
                    if geojson_path:
                        with open(geojson_path, 'r', encoding='utf-8') as f:
                            geojson_content = f.read()
                        
                        st.download_button(
                            label="📥 Скачать GeoJSON",
                            data=geojson_content,
                            file_name=f"{st.session_state.get('current_file', 'data')}.geojson",
                            mime="application/json"
                        )

def show_geo_analysis():
    """Показать вкладку геоанализа"""
    st.header("🌍 Расширенная геоаналитика")
    
    gdf = st.session_state.get('current_gdf')
    df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    
    # Выбор инструментов анализа
    analysis_tool = st.selectbox(
        "Выберите инструмент анализа",
        ["Хороплетная карта", "Тепловая карта плотности", "Диаграмма Вороного", 
         "Пространственная кластеризация", "Поиск ближайших точек"]
    )
    
    if analysis_tool == "Хороплетная карта":
        st.subheader("🗺️ Хороплетная карта (Choropleth)")
        
        # Выбор колонки для значений
        value_options = df.select_dtypes(include=[np.number]).columns.tolist()
        if value_options:
            value_col = st.selectbox("Колонка для значений", value_options)
            
            # Выбор цветовой схемы
            fill_options = ['YlOrRd', 'YlOrBr', 'YlGnBu', 'YlGn', 'Reds', 'RdPu', 
                          'Purples', 'PuRd', 'PuBu', 'PuBuGn', 'OrRd', 'Oranges', 
                          'Greys', 'Greens', 'GnBu', 'BuPu', 'BuGn', 'Blues']
            fill_color = st.selectbox("Цветовая схема", fill_options)
            
            if st.button("Создать хороплетную карту"):
                choropleth_map = st.session_state.visualizer.create_choropleth_map(
                    gdf, value_col,
                    title=f"Хороплетная карта: {value_col}",
                    legend_name=value_col,
                    fill_color=fill_color
                )
                
                if choropleth_map:
                    from streamlit_folium import st_folium
                    st_folium(choropleth_map, width=800, height=600)
        else:
            st.warning("Нет числовых колонок для создания хороплетной карты")
    
    elif analysis_tool == "Тепловая карта плотности":
        st.subheader("🔥 Тепловая карта плотности точек")
        
        value_col = st.selectbox(
            "Колонка для веса (опционально)",
            ["Нет"] + df.select_dtypes(include=[np.number]).columns.tolist(),
            key="heatmap_value"
        )
        if value_col == "Нет":
            value_col = None
        
        # Настройки тепловой карты
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Радиус", 5, 50, 15)
            blur = st.slider("Размытие", 5, 50, 25)
        with col2:
            min_opacity = st.slider("Минимальная прозрачность", 0.0, 1.0, 0.4, 0.1)
            max_zoom = st.slider("Максимальный зум", 5, 20, 10)
        
        if st.button("Создать тепловую карту"):
            heatmap = st.session_state.visualizer.create_heatmap(
                gdf, value_col,
                radius=radius,
                blur=blur,
                min_opacity=min_opacity,
                max_zoom=max_zoom
            )
            if heatmap:
                from streamlit_folium import st_folium
                st_folium(heatmap, width=800, height=600)
    
    elif analysis_tool == "Диаграмма Вороного":
        st.subheader("📐 Диаграмма Вороного для пространственного разделения")
        
        buffer_percent = st.slider("Буферная зона (%)", 0, 50, 10)
        
        if st.button("Построить диаграмма Вороного"):
            voronoi_fig = st.session_state.visualizer.create_voronoi_diagram(
                gdf,
                buffer_percent=buffer_percent/100
            )
            if voronoi_fig:
                st.pyplot(voronoi_fig)
                
                st.markdown("""
                **Что такое диаграмма Вороного?**
                
                Диаграмма Вороного делит пространство на регионы так, что:
                - Каждый регион содержит одну точку
                - Все точки внутри региона ближе к своей точке, чем к любой другой
                
                **Применение:**
                - Определение зон обслуживания
                - Анализ покрытия
                - Пространственное планирование
                """)
    
    elif analysis_tool == "Пространственная кластеризация":
        st.subheader("🔢 Пространственная кластеризация точек")
        
        n_clusters = st.slider(
            "Количество кластеров",
            min_value=2,
            max_value=min(20, len(gdf)),
            value=min(5, len(gdf)),
            step=1
        )
        
        if st.button("Выполнить кластеризацию"):
            # Извлекаем координаты
            coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
            
            # Выполняем кластеризацию
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(coords)
            
            # Создаем DataFrame с результатами
            gdf_clustered = gdf.copy()
            gdf_clustered['cluster'] = clusters
            
            # Визуализация на карте
            st.subheader("Визуализация кластеров на карте")
            
            # Создаем карту с цветами по кластерам
            m = st.session_state.visualizer.create_base_map(gdf=gdf, height=600, width=800)
            m = st.session_state.visualizer.add_markers_to_map(
                m, gdf_clustered, gdf_clustered,
                color_by='cluster',
                cluster=False,
                color_palette='set1'
            )
            
            from streamlit_folium import st_folium
            st_folium(m, width=800, height=600)
            
            # Статистика по кластерам
            st.subheader("📊 Статистика по кластерам")
            cluster_stats = pd.DataFrame({
                'Кластер': range(n_clusters),
                'Количество точек': np.bincount(clusters),
                'Средняя широта': [gdf_clustered[gdf_clustered['cluster'] == i].geometry.y.mean() for i in range(n_clusters)],
                'Средняя долгота': [gdf_clustered[gdf_clustered['cluster'] == i].geometry.x.mean() for i in range(n_clusters)]
            })
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Центроиды кластеров
            st.subheader("📍 Центроиды кластеров")
            centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=['Долгота', 'Широта'])
            centroids_df.index.name = 'Кластер'
            st.dataframe(centroids_df, use_container_width=True)
    
    elif analysis_tool == "Поиск ближайших точек":
        st.subheader("🔍 Поиск ближайших точек")
        
        # Выбор целевой точки
        target_options = list(range(len(gdf)))
        target_display = [f"Точка {i}: ({gdf.iloc[i].geometry.y:.4f}, {gdf.iloc[i].geometry.x:.4f})" 
                         for i in range(min(50, len(gdf)))]
        
        target_idx = st.selectbox(
            "Выберите целевую точку",
            range(min(50, len(gdf))),
            format_func=lambda x: target_display[x]
        )
        
        n_neighbors = st.slider(
            "Количество ближайших точек",
            min_value=1,
            max_value=min(20, len(gdf)-1),
            value=5,
            step=1
        )
        
        if st.button("Найти ближайшие точки"):
            target_point = gdf.iloc[target_idx].geometry
            
            # Используем анализатор для поиска ближайших точек
            nearest = st.session_state.analyzer.find_nearest_points(gdf, target_point, n_neighbors)
            
            # Создаем карту
            m = st.session_state.visualizer.create_base_map(gdf=gdf, height=600, width=800)
            
            # Добавляем все точки
            m = st.session_state.visualizer.add_markers_to_map(
                m, gdf, df,
                cluster=False,
                color_palette='gray'
            )
            
            # Выделяем целевую точку
            folium.CircleMarker(
                location=[target_point.y, target_point.x],
                radius=15,
                popup=f"<b>Целевая точка</b><br>Индекс: {target_idx}",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.8,
                weight=3
            ).add_to(m)
            
            # Выделяем ближайшие точки
            for idx, row in nearest.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=10,
                    popup=f"<b>Ближайшая точка</b><br>Индекс: {idx}<br>Расстояние: {target_point.distance(row.geometry)*111:.2f} км",
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7,
                    weight=2
                ).add_to(m)
                
                # Добавляем линию к целевой точке
                folium.PolyLine(
                    locations=[[target_point.y, target_point.x], [row.geometry.y, row.geometry.x]],
                    color='blue',
                    weight=1,
                    opacity=0.5,
                    dash_array='5,5'
                ).add_to(m)
            
            from streamlit_folium import st_folium
            st_folium(m, width=800, height=600)
            
            # Таблица ближайших точек
            st.subheader("📋 Ближайшие точки")
            nearest_df = pd.DataFrame({
                'Индекс': nearest.index,
                'Широта': nearest.geometry.y,
                'Долгота': nearest.geometry.x,
                'Расстояние (км)': [target_point.distance(geom) * 111 for geom in nearest.geometry]
            })
            st.dataframe(nearest_df.sort_values('Расстояние (км)'), use_container_width=True)

def show_dashboard():
    """Показать вкладку дашборда"""
    st.header("📈 Интерактивный дашборд")
    
    # Получаем данные
    has_coords = st.session_state.get('has_coords', False)
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
    else:
        df = st.session_state.get('current_df')
    
    # Автоматическая визуализация
    st.subheader("🚀 Автоматическая визуализация")
    
    target_column = st.selectbox(
        "Выберите целевую переменную (опционально)",
        ["Нет"] + df.columns.tolist(),
        key="dashboard_target"
    )
    if target_column == "Нет":
        target_column = None
    
    if st.button("🤖 Создать автоматический дашборд", type="primary"):
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
                    st.plotly_chart(fig, use_container_width=True)
            
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
        st.info("Для корреляционного анализа нужно как минимум 2 числовых колонки")
    
    # Экспорт дашборда
    st.markdown("---")
    if st.button("💾 Сохранить дашборд как HTML", type="primary"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_path = Path("reports") / "dashboards" / f"дашборд_{st.session_state.get('current_file', 'data')}_{timestamp}.html"
        
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
        if st.button("🗺️ Создать карту", type="primary"):
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
        if st.button("🔥 Создать тепловую карту"):
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
        if st.button("💾 Экспорт в CSV"):
            csv_path = Path("reports") / "coords" / f"координаты_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            st.success(f"Данные сохранены: {csv_path}")
        
        if st.button("🌍 Экспорт в GeoJSON"):
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
        
        if st.button("🔢 Выполнить кластеризацию"):
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
    st.caption("🌍 Анализатор данных с геоаналитикой | Создано с помощью Streamlit, GeoPandas, Plotly")