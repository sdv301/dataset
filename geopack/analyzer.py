import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

class GeoDataAnalyzer:
    def __init__(self):
        self.data_folder = "data"
        self.reports_folder = "reports"
        self.geo_data_folder = "geo_data"
        self.current_gdf = None
        self.current_df = None
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
                gdf = gpd.read_file(file_path)
                if 'geometry' in gdf.columns:
                    gdf['latitude'] = gdf.geometry.y
                    gdf['longitude'] = gdf.geometry.x
                return gdf
            elif suffix == '.shp':
                gdf = gpd.read_file(file_path)
                if 'geometry' in gdf.columns:
                    gdf['latitude'] = gdf.geometry.y
                    gdf['longitude'] = gdf.geometry.x
                return gdf
            else:
                return None
        except Exception as e:
            import streamlit as st
            st.error(f"Ошибка загрузки файла {file_path.name}: {e}")
            return None
    
    def create_geodataframe(self, df, lat_col=None, lon_col=None):
        """Создать GeoDataFrame из обычного DataFrame"""
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            return gdf
        elif 'geometry' in df.columns:
            return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            return None
    
    def detect_coordinates(self, df):
        """Обнаружить колонки с координатами"""
        lat_cols = []
        lon_cols = []
        
        columns_lower = [str(col).lower() for col in df.columns]
        
        # Приоритет 1: точные совпадения с русскими названиями
        for i, col in enumerate(columns_lower):
            if col == 'широта':
                lat_cols.append(df.columns[i])
            if col == 'долгота':
                lon_cols.append(df.columns[i])
        
        # Приоритет 2: точные совпадения с английскими названиями
        if not lat_cols:
            for i, col in enumerate(columns_lower):
                if col == 'latitude' or col == 'lat':
                    lat_cols.append(df.columns[i])
        
        if not lon_cols:
            for i, col in enumerate(columns_lower):
                if col == 'longitude' or col == 'lon' or col == 'lng':
                    lon_cols.append(df.columns[i])
        
        # Приоритет 3: частичные совпадения
        if not lat_cols:
            for i, col in enumerate(columns_lower):
                if 'широт' in col or 'latitude' in col or 'lat' in col:
                    if col not in ['широта', 'latitude', 'lat']:
                        lat_cols.append(df.columns[i])
                        break
        
        if not lon_cols:
            for i, col in enumerate(columns_lower):
                if 'долгот' in col or 'longitude' in col or 'lon' in col or 'lng' in col:
                    if col not in ['долгота', 'longitude', 'lon', 'lng']:
                        lon_cols.append(df.columns[i])
                        break
        
        return lat_cols[:1], lon_cols[:1]
    
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
        
        bounds = gdf.total_bounds
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if area > 0:
            analysis['point_density'] = len(gdf) / area
        
        if len(gdf) > 10:
            coords = np.array([(geom.x, geom.y) for geom in gdf.geometry if geom])
            if len(coords) > 1:
                kmeans = KMeans(n_clusters=min(5, len(coords)))
                clusters = kmeans.fit_predict(coords)
                analysis['clusters'] = {
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_sizes': np.bincount(clusters).tolist()
                }
        
        return analysis
    
    def calculate_buffer_zones(self, gdf, distance_km=10):
        """Рассчитать буферные зоны вокруг точек"""
        if gdf is None:
            return None
        
        distance_deg = distance_km / 111
        gdf_with_buffer = gdf.copy()
        gdf_with_buffer['buffer_zone'] = gdf.geometry.buffer(distance_deg)
        
        return gdf_with_buffer
    
    def find_nearest_points(self, gdf, target_point, n=5):
        """Найти ближайшие точки к заданной"""
        if gdf is None:
            return None
        
        distances = gdf.geometry.distance(target_point)
        nearest_indices = distances.nsmallest(n + 1).index.tolist()
        
        if target_point in gdf.geometry.values:
            nearest_indices = nearest_indices[1:]
        
        return gdf.loc[nearest_indices]
    
    def export_to_geojson(self, gdf, filename):
        """Экспортировать GeoDataFrame в GeoJSON"""
        if gdf is not None:
            output_path = Path(self.reports_folder) / "geojson" / filename
            gdf.to_file(output_path, driver='GeoJSON')
            return output_path
        return None
    
    def get_column_statistics(self, df):
        """Получить статистику по колонкам"""
        col_info = pd.DataFrame({
            'Колонка': df.columns,
            'Тип': df.dtypes.astype(str),
            'Уникальных значений': df.nunique(),
            'Пропущено': df.isnull().sum(),
            '% Пропущено': (df.isnull().sum() / len(df) * 100).round(2)
        })
        return col_info
    
    def get_numeric_statistics(self, df):
        """Получить статистику по числовым колонкам"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return df[numeric_cols].describe()
        return pd.DataFrame()
    
    def detect_data_types(self, df):
        """Определить типы данных в DataFrame"""
        data_types = {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': df.select_dtypes(include=['datetime']).columns.tolist(),
            'boolean': df.select_dtypes(include=['bool']).columns.tolist()
        }
        return data_types
    
    def get_correlation_matrix(self, df):
        """Получить матрицу корреляций"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            return numeric_df.corr()
        return pd.DataFrame()
    
    def get_top_correlations(self, df, top_n=10):
        """Получить топ корреляций"""
        corr = self.get_correlation_matrix(df)
        if corr.empty:
            return pd.DataFrame()
        
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'Колонка 1': corr.columns[i],
                    'Колонка 2': corr.columns[j],
                    'Корреляция': abs(corr.iloc[i, j])
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        return corr_df.sort_values('Корреляция', ascending=False).head(top_n)
    
def convert_dms_to_decimal(self, coord_str):
    """
    Преобразовать координаты из формата N/E с градусами в десятичный формат.
    Пример: N67.45682409 -> 67.45682409, E153.71599186 -> 153.71599186
    """
    try:
        if pd.isna(coord_str):
            return None
        
        coord_str = str(coord_str).strip()
        
        # Убираем N/E/S/W и оставляем только число
        if coord_str[0] in ['N', 'S', 'E', 'W']:
            value = float(coord_str[1:])
            # Для южных широт и западных долгот делаем отрицательными
            if coord_str[0] in ['S', 'W']:
                value = -value
            return value
        else:
            # Если нет буквы, пытаемся преобразовать как число
            return float(coord_str)
    except Exception as e:
        return None