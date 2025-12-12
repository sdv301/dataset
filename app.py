import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import io
import base64
import json
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# Настройки страницы
st.set_page_config(
    page_title="🌍 Анализатор данных с геоаналитикой",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
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
    .geojson-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🌍 Анализатор данных с расширенной геоаналитикой")
st.markdown("---")

class GeoDataAnalyzer:
    def __init__(self):
        self.data_folder = "data"
        self.reports_folder = "reports"
        self.geo_data_folder = "geo_data"
        self.current_gdf = None  # Геодатафрейм
        self.current_df = None   # Обычный датафрейм
        self.setup_folders()
    
    def setup_folders(self):
        """Создать необходимые папки"""
        Path(self.data_folder).mkdir(exist_ok=True)
        Path(self.reports_folder).mkdir(exist_ok=True)
        Path(self.geo_data_folder).mkdir(exist_ok=True)
        Path(self.reports_folder).joinpath("visualizations").mkdir(exist_ok=True)
        Path(self.reports_folder).joinpath("dashboards").mkdir(exist_ok=True)
        Path(self.reports_folder).joinpath("geojson").mkdir(exist_ok=True)
        Path(self.reports_folder).joinpath("shapefiles").mkdir(exist_ok=True)
    
    def get_data_files(self):
        """Получить список файлов в папке data"""
        supported_ext = ['.csv', '.xlsx', '.xls', '.parquet', '.json', '.geojson', '.shp']
        files = []
        
        for ext in supported_ext:
            files.extend(list(Path(self.data_folder).glob(f"*{ext}")))
        
        return files
    
    def load_file(self, file_path):
        """Загрузить файл с поддержкой геоформатов"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif suffix == '.json':
                return pd.read_json(file_path)
            elif suffix == '.geojson':
                # Загружаем как GeoDataFrame
                gdf = gpd.read_file(file_path)
                # Конвертируем в обычный DataFrame с координатами
                if 'geometry' in gdf.columns:
                    gdf['latitude'] = gdf.geometry.y
                    gdf['longitude'] = gdf.geometry.x
                return gdf
            elif suffix == '.shp':
                # Загружаем shapefile
                gdf = gpd.read_file(file_path)
                if 'geometry' in gdf.columns:
                    gdf['latitude'] = gdf.geometry.y
                    gdf['longitude'] = gdf.geometry.x
                return gdf
            else:
                return None
        except Exception as e:
            st.error(f"Ошибка загрузки файла {file_path.name}: {e}")
            return None
    
    def create_geodataframe(self, df, lat_col=None, lon_col=None):
        """Создать GeoDataFrame из обычного DataFrame"""
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            # Преобразуем координаты в геометрию
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            return gdf
        elif 'geometry' in df.columns:
            # Уже есть геометрия
            return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            return None
    
    def detect_coordinates(self, df):
        """Обнаружить колонки с координатами (расширенная версия)"""
        lat_patterns = [
            'lat', 'latitude', 'широта', 'y', 'coord_y', 
            'y_coord', 'гео_широта', 'geo_lat'
        ]
        lon_patterns = [
            'lon', 'longitude', 'lng', 'долгота', 'x', 
            'coord_x', 'x_coord', 'гео_долгота', 'geo_lon'
        ]
        
        lat_cols = []
        lon_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            for pattern in lat_patterns:
                if pattern in col_lower:
                    lat_cols.append(col)
                    break
            for pattern in lon_patterns:
                if pattern in col_lower:
                    lon_cols.append(col)
                    break
        
        return lat_cols[:1], lon_cols[:1]  # Возвращаем первые найденные
    
    def spatial_analysis(self, gdf):
        """Пространственный анализ геоданных"""
        if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
            return None
        
        analysis = {
            'total_points': len(gdf),
            'bounds': gdf.total_bounds.tolist(),
            'area': gdf.unary_union.convex_hull.area,
            'centroid': gdf.unary_union.centroid.coords[0],
            'crs': str(gdf.crs),
            'geometry_types': gdf.geometry.type.unique().tolist()
        }
        
        # Вычисляем плотность точек
        bounds = gdf.total_bounds
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if area > 0:
            analysis['point_density'] = len(gdf) / area
        
        # Кластеризация точек
        if len(gdf) > 10:
            coords = np.array([(geom.x, geom.y) for geom in gdf.geometry if geom])
            if len(coords) > 1:
                # Простая пространственная кластеризация
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(5, len(coords)))
                clusters = kmeans.fit_predict(coords)
                analysis['clusters'] = {
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_sizes': np.bincount(clusters).tolist()
                }
        
        return analysis
    
    def create_heatmap(self, gdf, value_column=None):
        """Создать тепловую карту плотности точек"""
        if gdf is None:
            return None
        
        # Создаем базовую карту
        center_lat = gdf.geometry.y.mean()
        center_lon = gdf.geometry.x.mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Добавляем тепловую карту
        from folium.plugins import HeatMap
        
        if value_column and value_column in gdf.columns:
            # Взвешенная тепловая карта
            heat_data = [[row.geometry.y, row.geometry.x, row[value_column]] 
                        for _, row in gdf.iterrows()]
        else:
            # Обычная тепловая карта плотности
            heat_data = [[row.geometry.y, row.geometry.x] for _, row in gdf.iterrows()]
        
        HeatMap(heat_data).add_to(m)
        
        return m
    
    def create_voronoi_diagram(self, gdf):
        """Создать диаграмму Вороного для точек"""
        try:
            from scipy.spatial import Voronoi, voronoi_plot_2d
            import matplotlib.pyplot as plt
            
            points = np.array([(geom.x, geom.y) for geom in gdf.geometry])
            vor = Voronoi(points)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')
            ax.plot(points[:, 0], points[:, 1], 'ko', markersize=5)
            ax.set_title("Диаграмма Вороного")
            ax.set_xlabel("Долгота")
            ax.set_ylabel("Широта")
            
            return fig
        except Exception as e:
            st.warning(f"Не удалось создать диаграмму Вороного: {e}")
            return None
    
    def calculate_buffer_zones(self, gdf, distance_km=10):
        """Рассчитать буферные зоны вокруг точек"""
        if gdf is None:
            return None
        
        # Конвертируем расстояние из км в градусы (приблизительно)
        # 1 градус ≈ 111 км на экваторе
        distance_deg = distance_km / 111
        
        # Создаем буферные зоны
        gdf_with_buffer = gdf.copy()
        gdf_with_buffer['buffer_zone'] = gdf.geometry.buffer(distance_deg)
        
        return gdf_with_buffer
    
    def find_nearest_points(self, gdf, target_point, n=5):
        """Найти ближайшие точки к заданной"""
        if gdf is None:
            return None
        
        # Вычисляем расстояния
        distances = gdf.geometry.distance(target_point)
        
        # Находим ближайшие точки
        nearest_indices = distances.nsmallest(n + 1).index.tolist()
        if target_point in gdf.geometry:
            nearest_indices = nearest_indices[1:]  # Исключаем саму точку
        
        return gdf.loc[nearest_indices]
    
    def export_to_geojson(self, gdf, filename):
        """Экспортировать GeoDataFrame в GeoJSON"""
        if gdf is not None:
            output_path = Path(self.reports_folder) / "geojson" / filename
            gdf.to_file(output_path, driver='GeoJSON')
            return output_path
        return None

# Инициализация анализатора
analyzer = GeoDataAnalyzer()

# Сайдбар
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка файлов
    st.subheader("📁 Загрузка данных")
    
    # Способ 1: Из папки data
    files = analyzer.get_data_files()
    if files:
        selected_file = st.selectbox(
            "Выберите файл из папки 'data':",
            [f.name for f in files],
            index=0
        )
        
        if st.button("📥 Загрузить выбранный файл", type="primary"):
            file_path = Path(analyzer.data_folder) / selected_file
            df = analyzer.load_file(file_path)
            if df is not None:
                # Автоматическое определение координат
                lat_cols, lon_cols = analyzer.detect_coordinates(df)
                
                if lat_cols and lon_cols:
                    gdf = analyzer.create_geodataframe(df, lat_cols[0], lon_cols[0])
                    st.session_state['current_gdf'] = gdf
                    st.session_state['has_coords'] = True
                    st.session_state['lat_col'] = lat_cols[0]
                    st.session_state['lon_col'] = lon_cols[0]
                else:
                    st.session_state['current_df'] = df
                    st.session_state['has_coords'] = False
                
                st.session_state['current_file'] = selected_file
                st.success(f"Файл {selected_file} загружен успешно!")
    
    # Способ 2: Загрузка файла
    st.subheader("Или загрузите файл:")
    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=['csv', 'xlsx', 'xls', 'parquet', 'json', 'geojson', 'shp'],
        help="Поддерживаются CSV, Excel, Parquet, JSON, GeoJSON, Shapefile"
    )
    
    if uploaded_file is not None:
        try:
            # Сохраняем временный файл
            temp_path = Path("temp") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            df = analyzer.load_file(temp_path)
            
            if df is not None:
                # Определяем тип данных
                lat_cols, lon_cols = analyzer.detect_coordinates(df)
                
                if lat_cols and lon_cols:
                    gdf = analyzer.create_geodataframe(df, lat_cols[0], lon_cols[0])
                    st.session_state['current_gdf'] = gdf
                    st.session_state['has_coords'] = True
                    st.session_state['lat_col'] = lat_cols[0]
                    st.session_state['lon_col'] = lon_cols[0]
                elif isinstance(df, gpd.GeoDataFrame):
                    st.session_state['current_gdf'] = df
                    st.session_state['has_coords'] = True
                else:
                    st.session_state['current_df'] = df
                    st.session_state['has_coords'] = False
                
                st.session_state['current_file'] = uploaded_file.name
                st.success(f"Файл {uploaded_file.name} загружен успешно!")
                
                # Удаляем временный файл
                temp_path.unlink()
                
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    
    # Гео-инструменты
    st.markdown("---")
    st.subheader("🗺️ Геоинструменты")
    
    if st.session_state.get('has_coords', False):
        st.success("✅ Геоданные доступны")
        
        # Выбор системы координат
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

# Основное содержимое
if 'current_gdf' not in st.session_state and 'current_df' not in st.session_state:
    # Стартовая страница
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

else:
    # Данные загружены
    filename = st.session_state.get('current_file', 'Файл')
    has_coords = st.session_state.get('has_coords', False)
    
    # Определяем тип данных
    if has_coords and 'current_gdf' in st.session_state:
        gdf = st.session_state['current_gdf']
        df = gdf.drop(columns=['geometry']) if 'geometry' in gdf.columns else gdf
        data_type = "🌍 Геоданные"
    else:
        df = st.session_state.get('current_df')
        gdf = None
        data_type = "📊 Обычные данные"
    
    # Создаем вкладки
    tabs = ["📋 Обзор данных", "📊 Визуализация"]
    
    if has_coords:
        tabs.extend(["🗺️ Карта", "🌍 Геоаналитика", "📈 Дашборд"])
    else:
        tabs.extend(["📈 Дашборд"])
    
    tab_objects = st.tabs(tabs)
    
    # Вкладка 1: Обзор данных
    with tab_objects[0]:
        st.header(f"{data_type} - Обзор")
        
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
                st.metric("Числовых колонок", len(df.select_dtypes(include=[np.number]).columns))
        
        # Предпросмотр данных
        st.subheader("Предпросмотр данных")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Информация о колонках
        st.subheader("Информация о колонках")
        col_info = pd.DataFrame({
            'Колонка': df.columns,
            'Тип': df.dtypes.astype(str),
            'Уникальных значений': df.nunique(),
            'Пропущено': df.isnull().sum(),
            '% Пропущено': (df.isnull().sum() / len(df) * 100).round(2)
        })
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Вкладка 2: Визуализация
    with tab_objects[1]:
        st.header("📊 Интерактивная визуализация")
        
        col1, col2 = st.columns(2)
        
        with col1:
            graph_type = st.selectbox(
                "Тип графика",
                ["Точечный график", "Линейный график", "Гистограмма", 
                 "Столбчатая диаграмма", "Круговая диаграмма", "Ящик с усами",
                 "Тепловая карта"],
                key="viz_type"
            )
            
            x_col = st.selectbox("Ось X", df.columns.tolist(), key="viz_x")
            
            if graph_type not in ["Гистограмма", "Круговая диаграмма", "Тепловая карта"]:
                y_col = st.selectbox("Ось Y", df.columns.tolist(), key="viz_y")
            else:
                y_col = None
        
        with col2:
            if graph_type not in ["Гистограмма", "Круговая диаграмма"]:
                color_options = ["Нет"] + df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_col = st.selectbox("Цвет", color_options, key="viz_color")
                if color_col == "Нет":
                    color_col = None
            else:
                color_col = None
            
            if graph_type == "Тепловая карта":
                value_col = st.selectbox(
                    "Значения для тепловой карты",
                    ["Нет"] + df.select_dtypes(include=[np.number]).columns.tolist(),
                    key="heatmap_val"
                )
                if value_col == "Нет":
                    value_col = None
            
            plot_title = st.text_input("Название графика", value="График", key="viz_title")
        
        if st.button("🔄 Создать график", type="primary", key="create_viz"):
            try:
                fig = None
                
                if graph_type == "Точечный график":
                    if color_col:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                       title=plot_title, hover_data=df.columns.tolist())
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col,
                                       title=plot_title, hover_data=df.columns.tolist())
                
                elif graph_type == "Линейный график":
                    fig = px.line(df, x=x_col, y=y_col,
                                title=plot_title)
                
                elif graph_type == "Гистограмма":
                    fig = px.histogram(df, x=x_col, color=color_col,
                                     title=plot_title, nbins=30)
                
                elif graph_type == "Столбчатая диаграмма":
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                               title=plot_title)
                
                elif graph_type == "Круговая диаграмма":
                    if df[x_col].nunique() > 20:
                        values = df[x_col].value_counts().head(10)
                        fig = px.pie(names=values.index, values=values.values,
                                   title=f"{plot_title} (первые 10 значений)")
                    else:
                        values = df[x_col].value_counts()
                        fig = px.pie(names=values.index, values=values.values,
                                   title=plot_title)
                
                elif graph_type == "Ящик с усами":
                    fig = px.box(df, x=color_col if color_col else x_col, y=y_col,
                               title=plot_title)
                
                elif graph_type == "Тепловая карта":
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        fig = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdBu',
                            zmid=0,
                            text=corr.round(2).values,
                            texttemplate='%{text}'
                        ))
                        fig.update_layout(title=plot_title, height=600)
                
                if fig:
                    fig.update_layout(height=600, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Скачать график
                    buf = io.BytesIO()
                    fig.write_html(buf, include_plotlyjs='cdn')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="график.html">💾 Скачать график</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    # Вкладка 3: Карта (только для геоданных)
    if has_coords and len(tabs) > 2:
        with tab_objects[2]:
            st.header("🗺️ Интерактивная карта")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Настройки карты
                map_type = st.selectbox(
                    "Тип карты",
                    ["OpenStreetMap", "Stamen Terrain", "Stamen Toner", "CartoDB positron"],
                    key="map_type"
                )
                
                # Выбор колонки для цвета
                color_options = ["Нет"] + df.select_dtypes(include=[np.number]).columns.tolist()
                color_by = st.selectbox("Цвет по колонке", color_options, key="map_color")
                
                # Создание карты
                if gdf is not None:
                    center_lat = gdf.geometry.y.mean()
                    center_lon = gdf.geometry.x.mean()
                    
                    m = folium.Map(location=[center_lat, center_lon], 
                                 zoom_start=10,
                                 tiles=map_type)
                    
                    # Добавление маркеров
                    for idx, row in gdf.iterrows():
                        popup_text = f"<b>Объект {idx}</b><br>"
                        for col in df.columns[:5]:
                            popup_text += f"{col}: {row[col]}<br>"
                        
                        # Цвет маркера
                        if color_by != "Нет" and color_by in row:
                            color_val = row[color_by]
                            # Нормализация для цвета
                            norm_val = (color_val - df[color_by].min()) / (df[color_by].max() - df[color_by].min())
                            color = plt.cm.viridis(norm_val)
                            color_hex = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                        else:
                            color_hex = '#3388ff'
                        
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=8,
                            popup=popup_text,
                            color=color_hex,
                            fill=True,
                            fill_opacity=0.7
                        ).add_to(m)
                    
                    # Отображение карты
                    st_folium(m, width=800, height=600)
            
            with col2:
                st.subheader("Инструменты карты")
                
                # Экспорт карты
                if st.button("💾 Сохранить карту как HTML"):
                    if gdf is not None:
                        map_path = Path(analyzer.reports_folder) / "visualizations" / f"карта_{filename}.html"
                        m.save(str(map_path))
                        st.success(f"Карта сохранена: {map_path}")
                
                # Экспорт в GeoJSON
                if st.button("🌍 Экспорт в GeoJSON"):
                    if gdf is not None:
                        geojson_path = analyzer.export_to_geojson(gdf, f"{filename}.geojson")
                        if geojson_path:
                            st.success(f"GeoJSON сохранен: {geojson_path}")
                            
                            # Предпросмотр GeoJSON
                            with open(geojson_path, 'r', encoding='utf-8') as f:
                                geojson_data = json.load(f)
                            
                            st.download_button(
                                label="📥 Скачать GeoJSON",
                                data=json.dumps(geojson_data, ensure_ascii=False),
                                file_name=f"{filename}.geojson",
                                mime="application/json"
                            )
                
                # Пространственная информация
                if gdf is not None:
                    st.markdown("### Пространственная информация")
                    bounds = gdf.total_bounds
                    st.write(f"**Широта:** {bounds[1]:.4f} - {bounds[3]:.4f}")
                    st.write(f"**Долгота:** {bounds[0]:.4f} - {bounds[2]:.4f}")
                    st.write(f"**Всего точек:** {len(gdf)}")
                    st.write(f"**CRS:** {gdf.crs}")
    
    # Вкладка 4: Геоаналитика (только для геоданных)
    if has_coords and len(tabs) > 3:
        with tab_objects[3]:
            st.header("🌍 Расширенная геоаналитика")
            
            # Выбор инструментов анализа
            analysis_tool = st.selectbox(
                "Выберите инструмент анализа",
                ["Тепловая карта плотности", "Буферные зоны", "Диаграмма Вороного", 
                 "Пространственная кластеризация", "Поиск ближайших точек"]
            )
            
            if analysis_tool == "Тепловая карта плотности":
                st.subheader("🔥 Тепловая карта плотности точек")
                
                value_col = st.selectbox(
                    "Колонка для веса (опционально)",
                    ["Нет"] + df.select_dtypes(include=[np.number]).columns.tolist()
                )
                if value_col == "Нет":
                    value_col = None
                
                if st.button("Создать тепловую карту"):
                    heatmap = analyzer.create_heatmap(gdf, value_col)
                    if heatmap:
                        st_folium(heatmap, width=800, height=600)
            
            elif analysis_tool == "Буферные зоны":
                st.subheader("🔄 Буферные зоны вокруг точек")
                
                buffer_distance = st.slider(
                    "Радиус буфера (км)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1
                )
                
                if st.button("Создать буферные зоны"):
                    buffered_gdf = analyzer.calculate_buffer_zones(gdf, buffer_distance)
                    
                    if buffered_gdf is not None:
                        # Создаем карту с буферами
                        center_lat = buffered_gdf.geometry.y.mean()
                        center_lon = buffered_gdf.geometry.x.mean()
                        
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        # Добавляем буферные зоны
                        for _, row in buffered_gdf.iterrows():
                            # Точка
                            folium.CircleMarker(
                                location=[row.geometry.y, row.geometry.x],
                                radius=5,
                                color='blue',
                                fill=True
                            ).add_to(m)
                            
                            # Буферная зона
                            folium.GeoJson(
                                row['buffer_zone'],
                                style_function=lambda x: {
                                    'fillColor': 'orange',
                                    'color': 'orange',
                                    'fillOpacity': 0.3,
                                    'weight': 1
                                }
                            ).add_to(m)
                        
                        st_folium(m, width=800, height=600)
                        
                        st.info(f"Созданы буферные зоны радиусом {buffer_distance} км")
            
            elif analysis_tool == "Диаграмма Вороного":
                st.subheader("📐 Диаграмма Вороного для пространственного разделения")
                
                if st.button("Построить диаграмму Вороного"):
                    voronoi_fig = analyzer.create_voronoi_diagram(gdf)
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
                    max_value=min(10, len(gdf)),
                    value=min(5, len(gdf)),
                    step=1
                )
                
                if st.button("Выполнить кластеризацию"):
                    # Извлекаем координаты
                    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
                    
                    # K-means кластеризация
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(coords)
                    
                    # Добавляем кластеры к данным
                    gdf_clustered = gdf.copy()
                    gdf_clustered['cluster'] = clusters
                    
                    # Визуализация
                    fig = px.scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        color=clusters.astype(str),
                        title=f"Пространственная кластеризация ({n_clusters} кластеров)",
                        labels={'x': 'Долгота', 'y': 'Широта'}
                    )
                    
                    # Добавляем центры кластеров
                    fig.add_trace(go.Scatter(
                        x=kmeans.cluster_centers_[:, 0],
                        y=kmeans.cluster_centers_[:, 1],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='x'),
                        name='Центры кластеров'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Статистика по кластерам
                    cluster_stats = pd.DataFrame({
                        'Кластер': range(n_clusters),
                        'Количество точек': np.bincount(clusters),
                        'Средняя широта': [gdf_clustered[gdf_clustered['cluster'] == i].geometry.y.mean() for i in range(n_clusters)],
                        'Средняя долгота': [gdf_clustered[gdf_clustered['cluster'] == i].geometry.x.mean() for i in range(n_clusters)]
                    })
                    
                    st.dataframe(cluster_stats)
            
            elif analysis_tool == "Поиск ближайших точек":
                st.subheader("🔍 Поиск ближайших точек")
                
                # Выбор целевой точки
                target_idx = st.selectbox(
                    "Выберите целевую точку",
                    range(len(gdf)),
                    format_func=lambda x: f"Точка {x}: ({gdf.iloc[x].geometry.y:.4f}, {gdf.iloc[x].geometry.x:.4f})"
                )
                
                n_neighbors = st.slider(
                    "Количество ближайших точек",
                    min_value=1,
                    max_value=min(10, len(gdf)-1),
                    value=5,
                    step=1
                )
                
                if st.button("Найти ближайшие точки"):
                    target_point = gdf.iloc[target_idx].geometry
                    nearest = analyzer.find_nearest_points(gdf, target_point, n_neighbors)
                    
                    # Создаем карту
                    m = folium.Map(
                        location=[target_point.y, target_point.x],
                        zoom_start=12
                    )
                    
                    # Целевая точка
                    folium.CircleMarker(
                        location=[target_point.y, target_point.x],
                        radius=10,
                        color='red',
                        fill=True,
                        popup="Целевая точка"
                    ).add_to(m)
                    
                    # Ближайшие точки
                    for idx, row in nearest.iterrows():
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=6,
                            color='green',
                            fill=True,
                            popup=f"Ближайшая точка {idx}"
                        ).add_to(m)
                        
                        # Линия к целевой точке
                        folium.PolyLine(
                            locations=[[target_point.y, target_point.x], [row.geometry.y, row.geometry.x]],
                            color='blue',
                            weight=1,
                            dash_array='5, 5'
                        ).add_to(m)
                    
                    st_folium(m, width=800, height=600)
                    
                    # Таблица ближайших точек
                    st.subheader("Ближайшие точки")
                    nearest_df = pd.DataFrame({
                        'Индекс': nearest.index,
                        'Широта': nearest.geometry.y,
                        'Долгота': nearest.geometry.x,
                        'Расстояние (км)': [target_point.distance(geom) * 111 for geom in nearest.geometry]
                    })
                    st.dataframe(nearest_df.sort_values('Расстояние (км)'))
    
    # Вкладка 5: Дашборд
    dashboard_tab_idx = 4 if has_coords else 2
    with tab_objects[dashboard_tab_idx]:
        st.header("📈 Интерактивный дашборд")
        
        # Создаем сетку дашборда
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Ключевые метрики")
            
            # Динамические метрики
            metrics_cols = st.multiselect(
                "Выберите колонки для метрик",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                default=df.select_dtypes(include=[np.number]).columns.tolist()[:3]
            )
            
            for col in metrics_cols:
                if col in df.columns:
                    val1, val2 = st.columns(2)
                    with val1:
                        st.metric(f"Среднее {col}", f"{df[col].mean():.2f}")
                    with val2:
                        st.metric(f"Медиана {col}", f"{df[col].median():.2f}")
            
            # Круговая диаграмма
            if len(df.select_dtypes(include=['object']).columns) > 0:
                cat_col = st.selectbox(
                    "Категория для анализа",
                    df.select_dtypes(include=['object']).columns.tolist(),
                    key="dashboard_pie"
                )
                
                if df[cat_col].nunique() <= 15:
                    fig = px.pie(df, names=cat_col, 
                               title=f"Распределение по {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Тренды и распределения")
            
            # Выбор колонки для анализа
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                analysis_col = st.selectbox(
                    "Колонка для анализа",
                    df.select_dtypes(include=[np.number]).columns.tolist(),
                    key="dashboard_hist"
                )
                
                # Гистограмма
                fig = px.histogram(df, x=analysis_col, nbins=30,
                                 title=f"Распределение {analysis_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig2 = px.box(df, y=analysis_col,
                            title=f"Box plot {analysis_col}")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Матрица корреляций
        st.subheader("🌡️ Корреляционная матрица")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr.round(2).values,
                texttemplate='%{text}'
            ))
            
            fig.update_layout(
                title="Тепловая карта корреляций",
                height=500,
                xaxis_title="Колонки",
                yaxis_title="Колонки"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Самые сильные корреляции
            st.subheader("🔗 Самые сильные корреляции")
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append({
                        'Колонка 1': corr.columns[i],
                        'Колонка 2': corr.columns[j],
                        'Корреляция': abs(corr.iloc[i, j])
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Корреляция', ascending=False).head(10)
            st.dataframe(corr_df, use_container_width=True)
        
        # Экспорт дашборда
        st.markdown("---")
        if st.button("💾 Сохранить дашборд как HTML", type="primary"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_path = Path(analyzer.reports_folder) / "dashboards" / f"дашборд_{filename}_{timestamp}.html"
            
            # Создаем простой HTML отчет
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Дашборд анализа данных - {filename}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 10px; }}
                    .metric {{ background-color: #e7f3fe; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📊 Дашборд анализа данных</h1>
                    <p><strong>Файл:</strong> {filename}</p>
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

# Футер
st.markdown("---")
st.caption("🌍 Анализатор данных с геоаналитикой | Создано с помощью Streamlit, GeoPandas, Plotly")

# Информация о загруженных данных
if 'current_file' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Загружен файл:** {st.session_state['current_file']}
    
    **Тип данных:** {data_type}
    
    **Координаты:** {'✅ Доступны' if has_coords else '❌ Не обнаружены'}
    
    **Время загрузки:** {datetime.now().strftime('%H:%M:%S')}
    """)
    