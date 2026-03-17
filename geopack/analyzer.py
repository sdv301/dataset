# analyzer.py - исправленная версия
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from pathlib import Path
import warnings
# warnings.filterwarnings('ignore') # Removed global warnings suppression

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
                # Пробуем разные кодировки для CSV
                encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'latin1', 'windows-1251']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        # Пробуем преобразовать заголовок если в первой строке данные
                        if len(df.columns) == 1 and ';' in str(df.columns[0]):
                            # Файл с разделителем ;
                            df = pd.read_csv(file_path, encoding=encoding, sep=';')
                        elif len(df.columns) == 1 and ',' not in str(df.iloc[0, 0]):
                            # Возможно BOM или другие проблемы
                            df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                        
                        # Проверяем наличие данных
                        if df is not None and not df.empty:
                            # Обрабатываем специальный формат координат
                            df = self._process_special_coordinates(df)
                            break
                    except Exception as e:
                        print(f"Попытка кодировки {encoding} не удалась: {e}")
                        continue
                
                if df is None or df.empty:
                    return None
                
                return df
                
            elif suffix in ['.xlsx', '.xls']:
                # Загружаем только первый лист
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) == 0:
                    return None
                
                # Загружаем первый лист
                df = pd.read_excel(file_path, sheet_name=sheet_names[0])
                
                if df is not None and not df.empty:
                    # Обрабатываем специальный формат координат
                    df = self._process_special_coordinates(df)
                
                return df
                
            elif suffix == '.parquet':
                df = pd.read_parquet(file_path)
                if df is not None and not df.empty:
                    df = self._process_special_coordinates(df)
                return df
                
            elif suffix == '.json':
                df = pd.read_json(file_path)
                if df is not None and not df.empty:
                    df = self._process_special_coordinates(df)
                return df
                
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
            # Возвращаем None, ошибку покажем в UI
            print(f"Ошибка загрузки файла {file_path.name}: {e}")
            return None
    
    def _process_special_coordinates(self, df):
        """Обработать специальные форматы координат в DataFrame"""
        if df is None or df.empty:
            return df
        
        df_processed = df.copy()
        
        # Ищем колонки с координатами
        lat_cols = []
        lon_cols = []
        
        for col in df_processed.columns:
            col_lower = str(col).lower()
            
            # Ищем широту
            if any(keyword in col_lower for keyword in ['широт', 'lat', 'latitude']):
                lat_cols.append(col)
            
            # Ищем долготу
            if any(keyword in col_lower for keyword in ['долгот', 'lon', 'longitude', 'lng']):
                lon_cols.append(col)
        
        # Если нашли координаты, преобразуем их
        if lat_cols and lon_cols:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            
            # Преобразуем координаты
            df_processed[f'{lat_col}_decimal'] = df_processed[lat_col].apply(self.parse_special_coordinate)
            df_processed[f'{lon_col}_decimal'] = df_processed[lon_col].apply(self.parse_special_coordinate)
        
        return df_processed
    
    def parse_special_coordinate(self, coord):
        """
        Парсит координаты из разных форматов:
        1. N56.77882594 или E105.75483468 (ваш формат)
        2. 56.77882594 (обычный числовой)
        3. 56°46'44" (градусы-минуты-секунды)
        """
        try:
            if pd.isna(coord):
                return None
            
            # Если уже число
            if isinstance(coord, (int, float)):
                return float(coord)
            
            coord_str = str(coord).strip()
            
            if not coord_str:
                return None
            
            # 1. Формат N56.77882594 или E105.75483468
            if coord_str[0] in ['N', 'S', 'E', 'W', 'С', 'Ю', 'В', 'З']:
                # Определяем направление
                direction = coord_str[0]
                
                # Извлекаем числовую часть
                # Убираем букву и возможные пробелы
                num_part = coord_str[1:].strip()
                
                # Если есть запятая, это может быть объединенная координата
                if ',' in num_part:
                    # Например: N56.77882594,E105.75483468
                    parts = num_part.split(',')
                    if len(parts) == 2:
                        # Это широта и долгота вместе
                        # Мы в этом методе обрабатываем только одну координату
                        # Так что берем первую часть
                        num_part = parts[0]
                
                try:
                    value = float(num_part)
                    
                    # Корректируем знак по направлению
                    if direction in ['S', 's', 'Ю', 'ю']:  # Южная широта
                        value = -abs(value)
                    elif direction in ['W', 'w', 'З', 'з']:  # Западная долгота
                        value = -abs(value)
                    elif direction in ['N', 'n', 'С', 'с']:  # Северная широта
                        value = abs(value)
                    elif direction in ['E', 'e', 'В', 'в']:  # Восточная долгота
                        value = abs(value)
                    
                    return value
                except ValueError:
                    return None
            
            # 2. Формат с разделителем, например: 56.77882594,105.75483468
            elif ',' in coord_str:
                parts = coord_str.split(',')
                try:
                    # Берем первую часть как число
                    return float(parts[0].strip())
                except Exception as e:
                    return None
            
            # 3. Попробуем просто преобразовать в число
            else:
                try:
                    return float(coord_str)
                except Exception as e:
                    return None
                
        except Exception as e:
            print(f"Ошибка парсинга координаты '{coord}': {e}")
            return None
    
    def load_excel_sheet(self, file_path, sheet_name):
        """Загрузить конкретный лист Excel файла"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if df is not None and not df.empty:
                df = self._process_special_coordinates(df)
            return df
        except Exception as e:
            print(f"Ошибка загрузки листа {sheet_name}: {e}")
            return None
    
    def get_excel_sheets(self, file_path):
        """Получить список листов Excel файла"""
        try:
            excel_file = pd.ExcelFile(file_path)
            return excel_file.sheet_names
        except Exception as e:
            print(f"Ошибка чтения Excel файла {file_path}: {e}")
            return []
    
    def create_geodataframe(self, df, lat_col=None, lon_col=None):
        """Создать GeoDataFrame из обычного DataFrame"""
        if df is None or df.empty:
            return None
        
        # Если не указаны колонки, пытаемся определить автоматически
        if lat_col is None or lon_col is None:
            lat_cols, lon_cols = self.detect_coordinates(df)
            if lat_cols and lon_cols:
                lat_col = lat_cols[0]
                lon_col = lon_cols[0]
            else:
                return None
        
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            try:
                # Создаем копию для преобразования
                df_copy = df.copy()
                
                # Ищем преобразованные координаты
                lat_decimal_col = f"{lat_col}_decimal"
                lon_decimal_col = f"{lon_col}_decimal"
                
                # Если есть уже преобразованные координаты, используем их
                if lat_decimal_col in df_copy.columns and lon_decimal_col in df_copy.columns:
                    lat_data = df_copy[lat_decimal_col]
                    lon_data = df_copy[lon_decimal_col]
                else:
                    # Иначе преобразуем сами
                    lat_data = df_copy[lat_col].apply(self.parse_special_coordinate)
                    lon_data = df_copy[lon_col].apply(self.parse_special_coordinate)
                
                # Удаляем строки с NaN в координатах
                valid_coords = lat_data.notna() & lon_data.notna()
                df_copy = df_copy[valid_coords].copy()
                
                if df_copy.empty:
                    return None
                
                # Создаем геометрию
                geometry = [Point(xy) for xy in zip(
                    lon_data[valid_coords], 
                    lat_data[valid_coords]
                )]
                
                gdf = gpd.GeoDataFrame(df_copy, geometry=geometry, crs="EPSG:4326")
                return gdf
                
            except Exception as e:
                print(f"Ошибка создания GeoDataFrame: {e}")
                return None
        elif 'geometry' in df.columns:
            return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            return None
    
    def detect_coordinates(self, df):
        """Обнаружить колонки с координатами"""
        if df is None or df.empty:
            return [], []
            
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
    
    # Остальные методы остаются без изменений...
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
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
        """Преобразовать координаты из градусов-минут-секунд в десятичные"""
        try:
            if pd.isna(coord_str):
                return None
            
            coord_str = str(coord_str).strip()
            
            if not coord_str:
                return None
            
            # Уже числовой формат
            if coord_str[0] in ['N', 'S', 'E', 'W']:
                try:
                    value = float(coord_str[1:])
                    if coord_str[0] in ['S', 'W']:
                        value = -value
                    return value
                except ValueError:
                    return None
            else:
                try:
                    return float(coord_str)
                except ValueError:
                    return None
        except Exception as e:
            return None
    
    def detect_date_columns(self, df):
        """Обнаружить колонки с датами"""
        date_cols = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            else:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['дата', 'date', 'time', 'время', 'год', 'месяц', 'день']):
                    try:
                        test_series = pd.to_datetime(df[col], errors='coerce', format='mixed')
                        if test_series.notna().any():
                            date_cols.append(col)
                    except Exception as e:
                        continue
        
        return date_cols
    
    def detect_and_convert_dtypes(self, df):
        """Автоматически определить и преобразовать типы данных"""
        if df is None or df.empty:
            return df
        
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                continue
                
            col_lower = str(col).lower()
            is_date_like = any(word in col_lower for word in ['дата', 'date', 'time', 'время', 'год', 'месяц', 'день'])
            
            if is_date_like or df_clean[col].dtype == 'object':
                try:
                    converted = pd.to_datetime(df_clean[col], errors='coerce', format='mixed')
                    if converted.notna().sum() > len(df_clean) * 0.5:
                        df_clean[col] = converted
                except Exception as e:
                    pass
            
            elif df_clean[col].dtype == 'object':
                try:
                    numeric = pd.to_numeric(df_clean[col], errors='coerce')
                    if numeric.notna().sum() > len(df_clean) * 0.5 and 'id' not in col_lower:
                        df_clean[col] = numeric
                except Exception as e:
                    pass
        
        return df_clean