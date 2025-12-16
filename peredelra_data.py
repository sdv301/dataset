import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultiSheetDataCleaner:
    """Очиститель данных с поддержкой многолистовых Excel файлов"""
    
    def __init__(self, file_path):
        """
        Инициализация очистителя данных
        
        Args:
            file_path: путь к Excel файлу
        """
        self.file_path = Path(file_path)
        self.excel_data = None
        self.sheet_data = {}  # {sheet_name: DataFrame}
        self.original_sheet_data = {}  # оригинальные данные
        self.transformations_log = {}
        self.data_type = self.detect_file_type()
    
    def detect_file_type(self):
        """Определение типа файла"""
        suffix = self.file_path.suffix.lower()
        
        if suffix in ['.xlsx', '.xls', '.xlsm']:
            return 'excel'
        elif suffix == '.csv':
            return 'csv'
        else:
            return None
    
    def load_all_sheets(self):
        """
        Загружает все листы из Excel файла
        
        Returns:
            Словарь {имя_листа: DataFrame}
        """
        if self.data_type != 'excel':
            print("Файл не является Excel. Используйте метод load_data() для CSV.")
            return None
        
        try:
            self.excel_data = pd.ExcelFile(self.file_path)
            sheet_names = self.excel_data.sheet_names
            
            print(f"Файл: {self.file_path.name}")
            print(f"Найдено листов: {len(sheet_names)}")
            print("-" * 50)
            
            for sheet_name in sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                self.sheet_data[sheet_name] = df
                self.original_sheet_data[sheet_name] = df.copy()
                
                # Инициализируем лог для листа
                self.transformations_log[sheet_name] = []
                
                self._log_transformation(
                    sheet_name, 
                    f"Загружен лист: {sheet_name} ({len(df)} строк, {len(df.columns)} колонок)"
                )
                
                print(f"  ✓ {sheet_name}: {len(df)} строк, {len(df.columns)} колонок")
            
            print("-" * 50)
            return self.sheet_data
            
        except Exception as e:
            print(f"Ошибка загрузки Excel файла: {e}")
            return None
    
    def load_data(self, sheet_name=None, encoding='utf-8-sig'):
        """
        Загрузка данных (для CSV или одного листа Excel)
        
        Args:
            sheet_name: имя листа (для Excel)
            encoding: кодировка (для CSV)
            
        Returns:
            DataFrame с загруженными данными
        """
        if self.data_type == 'csv':
            try:
                df = pd.read_csv(self.file_path, encoding=encoding)
                self.sheet_data['CSV'] = df
                self.original_sheet_data['CSV'] = df.copy()
                self.transformations_log['CSV'] = []
                
                self._log_transformation(
                    'CSV',
                    f"Загружен CSV файл: {len(df)} строк, {len(df.columns)} колонок"
                )
                
                print(f"✓ CSV загружен: {len(df)} строк, {len(df.columns)} колонок")
                return df
                
            except Exception as e:
                print(f"Ошибка загрузки CSV: {e}")
                return None
                
        elif self.data_type == 'excel':
            return self.load_all_sheets()
    
    def _log_transformation(self, sheet_name, message):
        """Логирование преобразований для конкретного листа"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp}: {message}"
        self.transformations_log[sheet_name].append(log_entry)
    
    def clean_all_sheets(self, operations=None):
        """
        Очистка всех листов с указанными операциями
        
        Args:
            operations: список операций или None для всех
        """
        if not self.sheet_data:
            print("Данные не загружены")
            return
        
        # Операции по умолчанию
        if operations is None:
            operations = [
                'clean_names',
                'merge_dates',
                'parse_coords',
                'handle_missing',
                'detect_types'
            ]
        
        print("\n" + "=" * 60)
        print("НАЧАЛО ОЧИСТКИ ВСЕХ ЛИСТОВ")
        print("=" * 60)
        
        for sheet_name in self.sheet_data.keys():
            print(f"\nОбработка листа: {sheet_name}")
            self._clean_single_sheet(sheet_name, operations)
        
        print("\n" + "=" * 60)
        print("ОЧИСТКА ЗАВЕРШЕНА")
        print("=" * 60)
    
    def _clean_single_sheet(self, sheet_name, operations):
        """Очистка одного листа"""
        df = self.sheet_data[sheet_name]
        
        print(f"  Исходно: {len(df)} строк, {len(df.columns)} колонок")
        
        # Выполняем операции
        for op in operations:
            if op == 'clean_names':
                df = self._clean_column_names(df, sheet_name)
            elif op == 'merge_dates':
                df = self._merge_date_columns(df, sheet_name)
            elif op == 'parse_coords':
                df = self._parse_coordinates(df, sheet_name)
            elif op == 'handle_missing':
                df = self._handle_missing_values(df, sheet_name)
            elif op == 'detect_types':
                df = self._detect_data_types(df, sheet_name)
            elif op == 'convert_dates':
                df = self._convert_date_formats(df, sheet_name)
            elif op == 'extract_date_parts':
                df = self._extract_date_parts(df, sheet_name)
        
        # Обновляем данные
        self.sheet_data[sheet_name] = df
        
        print(f"  Результат: {len(df)} строк, {len(df.columns)} колонок")
    
    def _clean_column_names(self, df, sheet_name):
        """Очистка имен колонок"""
        original_cols = list(df.columns)
        
        new_columns = []
        for col in df.columns:
            new_name = str(col)
            
            # Удаляем специальные символы
            new_name = re.sub(r'[^\w\s]', '_', new_name)
            
            # Заменяем пробелы на подчеркивания
            new_name = re.sub(r'\s+', '_', new_name)
            
            # Убираем несколько подчеркиваний подряд
            new_name = re.sub(r'_+', '_', new_name)
            
            # Убираем подчеркивания в начале и конце
            new_name = new_name.strip('_')
            
            # Приводим к нижнему регистру
            new_name = new_name.lower()
            
            new_columns.append(new_name)
        
        df.columns = new_columns
        
        # Логируем изменения
        changed = []
        for old, new in zip(original_cols, new_columns):
            if old != new:
                changed.append((old, new))
        
        if changed:
            self._log_transformation(sheet_name, f"Очищены имена колонок: {len(changed)} изменений")
            if len(changed) <= 5:
                for old, new in changed:
                    self._log_transformation(sheet_name, f"  '{old}' -> '{new}'")
        
        return df
    
    def _merge_date_columns(self, df, sheet_name):
        """Объединение колонок даты (только дата, без времени)"""
        # Ищем колонки, связанные с датой
        date_patterns = {
            'год': ['год', 'year', 'yyyy', 'г.', 'year_'],
            'месяц': ['месяц', 'month', 'мм', 'мес.', 'month_'],
            'день': ['день', 'day', 'дд', 'дн.', 'day_'],
            'дата': ['дата', 'date', 'datetime', 'timestamp', 'дата_']
        }
        
        found_cols = {}
        
        # Ищем отдельные колонки год/месяц/день
        for col in df.columns:
            col_lower = str(col).lower()
            
            for date_type, patterns in date_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    found_cols[date_type] = col
                    break
        
        # Если нашли все три колонки (год, месяц, день)
        if all(k in found_cols for k in ['год', 'месяц', 'день']):
            try:
                # Создаем строки даты
                date_strings = []
                errors = 0
                
                for idx in df.index:
                    try:
                        # Берем значения
                        year_val = df.at[idx, found_cols['год']]
                        month_val = df.at[idx, found_cols['месяц']]
                        day_val = df.at[idx, found_cols['день']]
                        
                        # Пропускаем пустые значения
                        if pd.isna(year_val) or pd.isna(month_val) or pd.isna(day_val):
                            date_strings.append(None)
                            errors += 1
                            continue
                        
                        # Преобразуем в целые числа
                        try:
                            year_int = int(float(year_val))
                            month_int = int(float(month_val))
                            day_int = int(float(day_val))
                        except:
                            date_strings.append(None)
                            errors += 1
                            continue
                        
                        # Проверяем валидность даты
                        if (1 <= month_int <= 12 and 1 <= day_int <= 31 and 
                            1900 <= year_int <= 2100):
                            # Форматируем как строку даты без времени
                            date_str = f"{year_int:04d}-{month_int:02d}-{day_int:02d}"
                            date_strings.append(date_str)
                        else:
                            date_strings.append(None)
                            errors += 1
                            
                    except Exception as e:
                        date_strings.append(None)
                        errors += 1
                
                # Создаем колонку с датой (только дата, без времени)
                if date_strings:
                    # Преобразуем в datetime
                    df['дата'] = pd.to_datetime(date_strings, errors='coerce', format='%Y-%m-%d')
                    
                    # Если нужно, преобразуем в тип date (без времени)
                    # df['дата'] = df['дата'].dt.date
                
                # Удаляем исходные колонки
                cols_to_remove = [found_cols['год'], found_cols['месяц'], found_cols['день']]
                df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
                
                # Считаем успешно преобразованные
                valid_dates = df['дата'].notna().sum() if 'дата' in df.columns else 0
                
                self._log_transformation(
                    sheet_name, 
                    f"Объединены колонки даты: создана 'дата' ({valid_dates} валидных)"
                )
                
                if errors > 0:
                    self._log_transformation(sheet_name, f"  Не удалось преобразовать {errors} дат")
                
            except Exception as e:
                self._log_transformation(sheet_name, f"Ошибка объединения дат: {str(e)[:50]}")
        
        # Если нашли уже существующую колонку даты
        elif 'дата' in found_cols:
            date_col = found_cols['дата']
            try:
                # Преобразуем в datetime и убираем время
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                # Если нужно убрать время:
                # df[date_col] = df[date_col].dt.date
                
                valid_dates = df[date_col].notna().sum()
                self._log_transformation(
                    sheet_name,
                    f"Преобразована колонка даты '{date_col}' ({valid_dates} валидных)"
                )
            except:
                pass
        
        # Проверяем также форматы типа dd.mm.yyyy в других колонках
        return self._find_and_convert_date_formats(df, sheet_name)

    def _find_and_convert_date_formats(self, df, sheet_name):
        """Ищет и преобразует даты в форматах типа dd.mm.yyyy"""
        date_formats_to_try = [
            '%d.%m.%Y',      # 31.12.2023
            '%d-%m-%Y',      # 31-12-2023
            '%d/%m/%Y',      # 31/12/2023
            '%Y.%m.%d',      # 2023.12.31
            '%Y-%m-%d',      # 2023-12-31
            '%Y/%m/%d',      # 2023/12/31
            '%m.%d.%Y',      # 12.31.2023
            '%m-%d-%Y',      # 12-31-2023
            '%m/%d/%Y',      # 12/31/2023
        ]
        
        converted_cols = []
        
        for col in df.columns:
            # Пропускаем уже существующие колонки дат
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            # Проверяем первые непустые значения
            sample = df[col].dropna().head(10)
            if len(sample) < 3:
                continue
            
            # Пробуем определить, похожи ли значения на дату
            looks_like_date = False
            date_format = None
            
            for fmt in date_formats_to_try:
                try:
                    # Пробуем преобразовать несколько значений
                    test_vals = sample.head(3).astype(str).tolist()
                    success_count = 0
                    
                    for val in test_vals:
                        try:
                            pd.to_datetime(val, format=fmt, errors='raise')
                            success_count += 1
                        except:
                            pass
                    
                    if success_count >= 2:  # Если хотя бы 2 из 3 успешно
                        looks_like_date = True
                        date_format = fmt
                        break
                except:
                    continue
            
            if looks_like_date and date_format:
                try:
                    # Преобразуем всю колонку
                    original_name = col
                    df[f'{col}_дата'] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    
                    # Удаляем исходную колонку, если преобразование успешно
                    valid_count = df[f'{col}_дата'].notna().sum()
                    if valid_count > 0:
                        # Переименовываем обработанную колонку
                        df = df.rename(columns={f'{col}_дата': col})
                        
                        converted_cols.append((original_name, valid_count, date_format))
                        
                        self._log_transformation(
                            sheet_name,
                            f"Преобразована дата в колонке '{original_name}': формат {date_format} ({valid_count} записей)"
                        )
                    else:
                        # Удаляем неудачную попытку
                        df = df.drop(columns=[f'{col}_дата'])
                        
                except Exception as e:
                    pass
        
        if converted_cols:
            summary = ", ".join([f"{col} ({cnt})" for col, cnt, _ in converted_cols])
            self._log_transformation(sheet_name, f"Обнаружены и преобразованы даты: {summary}")
        
        return df
    
    def _extract_date_parts(self, df, sheet_name):
        """
        Извлечение частей даты из существующих колонок с датами (ТОЛЬКО ДАТА)
        
        Args:
            df: DataFrame для обработки
            sheet_name: имя листа для логирования
            
        Returns:
            Обработанный DataFrame
        """
        date_columns = self._find_date_columns_auto(df)
        
        if not date_columns:
            return df
        
        extracted_cols = []
        
        for date_col in date_columns:
            # Проверяем, что колонка действительно содержит даты
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                # Пробуем преобразовать
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                except:
                    continue
            
            # Извлекаем части даты
            try:
                # Если есть время, убираем его
                if hasattr(df[date_col].dt, 'date'):
                    df[date_col] = df[date_col].dt.date
                
                # Извлекаем год, месяц, день
                df[f'{date_col}_год'] = df[date_col].apply(lambda x: x.year if pd.notna(x) else np.nan)
                df[f'{date_col}_месяц'] = df[date_col].apply(lambda x: x.month if pd.notna(x) else np.nan)
                df[f'{date_col}_день'] = df[date_col].apply(lambda x: x.day if pd.notna(x) else np.nan)
                
                # Извлекаем день недели (на русском)
                def get_russian_weekday(date_val):
                    if pd.isna(date_val):
                        return np.nan
                    
                    # Маппинг дней недели
                    weekdays_ru = {
                        0: 'Понедельник',
                        1: 'Вторник',
                        2: 'Среда',
                        3: 'Четверг',
                        4: 'Пятница',
                        5: 'Суббота',
                        6: 'Воскресенье'
                    }
                    
                    # Получаем номер дня недели (0=Понедельник, 6=Воскресенье)
                    weekday_num = date_val.weekday()
                    return weekdays_ru.get(weekday_num, 'Неизвестно')
                
                df[f'{date_col}_день_недели'] = df[date_col].apply(get_russian_weekday)
                
                # Извлекаем квартал
                df[f'{date_col}_квартал'] = df[date_col].apply(lambda x: ((x.month - 1) // 3) + 1 if pd.notna(x) else np.nan)
                
                extracted_cols.append(date_col)
                
                self._log_transformation(
                    sheet_name,
                    f"Извлечены части даты из '{date_col}': год, месяц, день, день недели, квартал"
                )
                
            except Exception as e:
                self._log_transformation(sheet_name, f"Ошибка извлечения частей даты из '{date_col}': {str(e)[:50]}")
        
        return df
    
    def _find_date_columns_auto(self, df):
        """Автоматический поиск колонок с датами"""
        date_columns = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Проверяем по названию
            date_keywords = [
                'дата', 'date', 'время', 'time', 'год', 'месяц', 'день',
                'timestamp', 'datetime', 'срок', 'период', 'от', 'до',
                'начало', 'конец', 'start', 'end', 'created', 'updated'
            ]
            
            if any(keyword in col_lower for keyword in date_keywords):
                date_columns.append(col)
                continue
            
            # Проверяем по содержимому (первые 10 непустых значений)
            sample = df[col].dropna().head(10)
            if len(sample) == 0:
                continue
            
            # Пробуем распознать дату
            try:
                # Пробуем разные форматы
                test_vals = sample.head(3).astype(str).tolist()
                
                # Проверяем, похожи ли значения на дату
                date_patterns = [
                    r'\d{4}[-./]\d{1,2}[-./]\d{1,2}',  # 2023-12-31
                    r'\d{1,2}[-./]\d{1,2}[-./]\d{4}',  # 31-12-2023
                    r'\d{1,2}\s+[а-яa-z]+\s+\d{4}',    # 31 декабря 2023
                    r'[а-яa-z]+\s+\d{1,2},\s+\d{4}',   # December 31, 2023
                ]
                
                looks_like_date = False
                for val in test_vals:
                    for pattern in date_patterns:
                        if re.search(pattern, val, re.IGNORECASE):
                            looks_like_date = True
                            break
                    if looks_like_date:
                        break
                
                if looks_like_date:
                    date_columns.append(col)
            except:
                pass
        
        return list(set(date_columns))  # Убираем дубликаты
    
    def _parse_coordinates(self, df, sheet_name):
        """Преобразование координат"""
        # Ищем колонки с координатами
        lat_patterns = ['широта', 'latitude', 'lat', 'y', 'координата_y']
        lon_patterns = ['долгота', 'longitude', 'lon', 'lng', 'x', 'координата_x']
        
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            if not lat_col and any(pattern in col_lower for pattern in lat_patterns):
                lat_col = col
            
            if not lon_col and any(pattern in col_lower for pattern in lon_patterns):
                lon_col = col
        
        if lat_col and lon_col:
            try:
                # Функция для преобразования координат
                def parse_coord(value):
                    if pd.isna(value):
                        return np.nan
                    
                    value_str = str(value).strip().upper()
                    
                    # Пробуем как число
                    try:
                        return float(value_str)
                    except:
                        pass
                    
                    # Форматы типа N56.77882594
                    match = re.search(r'([NS]?)([+-]?\d+\.?\d*)([EW]?)', value_str)
                    if match:
                        prefix = match.group(1)
                        number = float(match.group(2))
                        suffix = match.group(3)
                        
                        if prefix in ['S', 'W'] or suffix in ['W']:
                            return -abs(number)
                        return abs(number)
                    
                    return np.nan
                
                # Применяем преобразование
                df[f'{lat_col}_число'] = df[lat_col].apply(parse_coord)
                df[f'{lon_col}_число'] = df[lon_col].apply(parse_coord)
                
                # Статистика
                lat_converted = df[f'{lat_col}_число'].notna().sum()
                lon_converted = df[f'{lon_col}_число'].notna().sum()
                
                self._log_transformation(
                    sheet_name,
                    f"Преобразованы координаты: {lat_col}->{lat_col}_число ({lat_converted}), {lon_col}->{lon_col}_число ({lon_converted})"
                )
                
            except Exception as e:
                self._log_transformation(sheet_name, f"Ошибка преобразования координат: {str(e)[:50]}")
        
        return df
    
    def _merge_date_columns_simple(self, df, sheet_name, keep_original=True):
        """Простое объединение колонок даты"""
        # Определяем какие колонки что содержат
        year_col = None
        month_col = None
        day_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            if 'год' in col_lower or 'year' in col_lower:
                year_col = col
            elif 'месяц' in col_lower or 'month' in col_lower:
                month_col = col
            elif 'день' in col_lower or 'day' in col_lower:
                day_col = col
        
        # Если нашли все три
        if year_col and month_col and day_col:
            try:
                # Создаем объединенную дату
                dates = []
                for idx in df.index:
                    y = df.at[idx, year_col]
                    m = df.at[idx, month_col]
                    d = df.at[idx, day_col]
                    
                    if pd.notna(y) and pd.notna(m) and pd.notna(d):
                        try:
                            # Форматируем как дату
                            date_str = f"{int(y)}-{int(m):02d}-{int(d):02d}"
                            dates.append(date_str)
                        except:
                            dates.append(None)
                    else:
                        dates.append(None)
                
                # Добавляем новую колонку
                df['дата'] = pd.to_datetime(dates, errors='coerce', format='%Y-%m-%d')
                
                # Убираем время, если оно добавилось
                if hasattr(df['дата'].dt, 'date'):
                    df['дата'] = df['дата'].dt.date
                
                # Удаляем исходные, если не нужно сохранять
                if not keep_original:
                    df = df.drop(columns=[year_col, month_col, day_col])
                
                self._log_transformation(
                    sheet_name,
                    f"Объединены даты: {year_col}, {month_col}, {day_col} → дата"
                )
                
            except Exception as e:
                self._log_transformation(sheet_name, f"Ошибка объединения дат: {str(e)[:50]}")
        
        return df
    
    def _handle_missing_values(self, df, sheet_name):
        """Обработка пропущенных значений"""
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            # Для числовых колонок
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col] = df[col].fillna(median_val)
            
            # Для текстовых колонок
            text_cols = df.select_dtypes(include=['object']).columns
            for col in text_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna('Не указано')
            
            # Для дат
            date_cols = df.select_dtypes(include=['datetime64']).columns
            for col in date_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(pd.NaT)
            
            missing_after = df.isnull().sum().sum()
            
            self._log_transformation(
                sheet_name,
                f"Обработаны пропущенные значения: {missing_before} -> {missing_after}"
            )
        
        return df
    
    def _detect_data_types(self, df, sheet_name):
        """Определение типов данных"""
        type_changes = []
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            # Пропускаем уже обработанные даты
            if 'datetime' in original_dtype:
                continue
            
            # Пробуем преобразовать в datetime
            if self._is_potential_date_column(col, df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if str(df[col].dtype) != original_dtype:
                        type_changes.append((col, original_dtype, 'datetime'))
                except:
                    pass
            
            # Пробуем преобразовать в число
            elif self._is_potential_numeric_column(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if str(df[col].dtype) != original_dtype:
                        type_changes.append((col, original_dtype, str(df[col].dtype)))
                except:
                    pass
            
            # Пробуем преобразовать в булевый тип
            elif self._is_potential_boolean_column(df[col]):
                try:
                    bool_map = {
                        'да': True, 'нет': False,
                        'true': True, 'false': False,
                        '1': True, '0': False,
                        'yes': True, 'no': False,
                        'вкл': True, 'выкл': False,
                        'on': True, 'off': False
                    }
                    
                    def to_bool(val):
                        if pd.isna(val):
                            return np.nan
                        val_str = str(val).lower().strip()
                        return bool_map.get(val_str, val)
                    
                    df[col] = df[col].apply(to_bool)
                    type_changes.append((col, original_dtype, 'bool'))
                except:
                    pass
        
        if type_changes:
            self._log_transformation(
                sheet_name,
                f"Определены типы данных: {len(type_changes)} изменений"
            )
            for col, old_type, new_type in type_changes[:3]:
                self._log_transformation(sheet_name, f"  {col}: {old_type} -> {new_type}")
        
        return df
    
    def _is_potential_date_column(self, col_name, series):
        """Проверяет, может ли колонка быть датой"""
        col_lower = str(col_name).lower()
        date_keywords = ['дата', 'date', 'время', 'time', 'год', 'месяц', 'день']
        
        if any(keyword in col_lower for keyword in date_keywords):
            return True
        
        # Проверяем первые значения
        sample = series.dropna().head(5)
        if len(sample) == 0:
            return False
        
        # Пробуем преобразовать
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False
    
    def _is_potential_numeric_column(self, series):
        """Проверяет, может ли колонка быть числовой"""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        # Пробуем преобразовать
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def _is_potential_boolean_column(self, series):
        """Проверяет, может ли колонка быть булевой"""
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        
        # Проверяем уникальные значения
        unique_vals = set(str(val).lower().strip() for val in sample)
        
        # Наборы булевых значений
        bool_sets = [
            {'да', 'нет', ''},
            {'true', 'false', ''},
            {'1', '0', ''},
            {'yes', 'no', ''},
            {'вкл', 'выкл', ''},
            {'on', 'off', ''}
        ]
        
        for bool_set in bool_sets:
            if unique_vals.issubset(bool_set):
                return True
        
        return False
    
    # Методы для работы с отдельными листами
    def convert_dates_in_sheet(self, sheet_name, date_formats=None):
        """
        Преобразование дат в конкретном листе
        
        Args:
            sheet_name: имя листа
            date_formats: список форматов для попытки преобразования
        """
        if sheet_name not in self.sheet_data:
            print(f"Лист '{sheet_name}' не найден")
            return
        
        df = self.sheet_data[sheet_name]
        df = self._convert_date_formats(df, sheet_name)
        self.sheet_data[sheet_name] = df
    
    def extract_date_parts_in_sheet(self, sheet_name):
        """Извлечение частей даты в конкретном листе"""
        if sheet_name not in self.sheet_data:
            print(f"Лист '{sheet_name}' не найден")
            return
        
        df = self.sheet_data[sheet_name]
        df = self._extract_date_parts(df, sheet_name)
        self.sheet_data[sheet_name] = df
    
    def save_to_excel(self, output_path=None):
        """
        Сохраняет все листы в один Excel файл
        
        Args:
            output_path: путь для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        if not self.sheet_data:
            print("Нет данных для сохранения")
            return None
        
        # Определяем путь для сохранения
        if output_path is None:
            # Сохраняем рядом с исходным файлом
            output_path = self.file_path.parent / f"{self.file_path.stem}_очищенный.xlsx"
        else:
            output_path = Path(output_path)
        
        try:
            # Проверяем/создаем директорию
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\nСохранение файла: {output_path.name}")
            print(f"Листов для сохранения: {len(self.sheet_data)}")
            
            # Создаем Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in self.sheet_data.items():
                    # Ограничиваем имя листа 31 символом (ограничение Excel)
                    excel_sheet_name = sheet_name[:31]
                    
                    # Сохраняем данные
                    df.to_excel(writer, sheet_name=excel_sheet_name, index=False)
                    
                    # Автонастройка ширины колонок
                    worksheet = writer.sheets[excel_sheet_name]
                    
                    for column in df:
                        column_width = max(
                            df[column].astype(str).map(len).max(),
                            len(str(column))
                        ) + 2
                        
                        col_idx = df.columns.get_loc(column)
                        worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_width, 50)
                    
                    print(f"  ✓ {sheet_name}: {len(df)} строк, {len(df.columns)} колонок")
            
            # Проверяем размер файла
            file_size = output_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"\n✓ Файл успешно сохранен:")
            print(f"  Путь: {output_path}")
            print(f"  Размер: {file_size_mb:.2f} MB")
            print(f"  Всего листов: {len(self.sheet_data)}")
            
            # Сохраняем лог отдельным файлом
            self._save_transformations_log(output_path.parent)
            
            return output_path
            
        except PermissionError:
            print(f"\n✗ Ошибка: Нет прав на запись в {output_path}")
            print("Попробуйте другой путь или закройте файл, если он открыт.")
            
            # Пробуем сохранить на рабочем столе
            desktop = Path.home() / "Desktop"
            alt_path = desktop / f"{self.file_path.stem}_очищенный.xlsx"
            
            retry = input(f"\nПопробовать сохранить на Рабочий стол ({alt_path})? (y/n): ").lower().strip()
            
            if retry == 'y':
                return self.save_to_excel(alt_path)
            else:
                return None
            
        except Exception as e:
            print(f"\n✗ Ошибка при сохранении: {e}")
            return None
    
    def save_to_csv(self, output_dir=None):
        """
        Сохраняет каждый лист в отдельный CSV файл
        
        Args:
            output_dir: директория для сохранения
            
        Returns:
            Список путей к сохраненным файлам
        """
        if not self.sheet_data:
            print("Нет данных для сохранения")
            return []
        
        # Определяем директорию для сохранения
        if output_dir is None:
            output_dir = self.file_path.parent / f"{self.file_path.stem}_csv"
        else:
            output_dir = Path(output_dir)
        
        # Создаем директорию
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        print(f"\nСохранение CSV файлов в: {output_dir}")
        
        for sheet_name, df in self.sheet_data.items():
            # Создаем имя файла
            safe_name = self._make_filename_safe(sheet_name)
            csv_path = output_dir / f"{safe_name}.csv"
            
            try:
                # Сохраняем в CSV
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                saved_files.append(csv_path)
                
                print(f"  ✓ {sheet_name} -> {csv_path.name} ({len(df)} строк)")
                
            except Exception as e:
                print(f"  ✗ Ошибка при сохранении {sheet_name}: {e}")
        
        # Сохраняем лог
        self._save_transformations_log(output_dir)
        
        if saved_files:
            print(f"\n✓ Сохранено файлов: {len(saved_files)}")
            return saved_files
        else:
            print("\n✗ Не удалось сохранить ни одного файла")
            return []
    
    def _save_transformations_log(self, output_dir):
        """Сохраняет лог преобразований"""
        if not self.transformations_log:
            return
        
        log_path = output_dir / f"{self.file_path.stem}_преобразования.log"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Лог преобразований файла: {self.file_path.name}\n")
            f.write(f"Время обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for sheet_name, logs in self.transformations_log.items():
                f.write(f"ЛИСТ: {sheet_name}\n")
                f.write("-" * 40 + "\n")
                
                if logs:
                    for log_entry in logs:
                        f.write(f"{log_entry}\n")
                else:
                    f.write("Нет преобразований\n")
                
                f.write("\n")
        
        print(f"  Лог преобразований: {log_path.name}")
    
    def _make_filename_safe(self, text):
        """Делает строку безопасной для имени файла"""
        # Заменяем недопустимые символы
        safe = re.sub(r'[\\/*?:"<>|]', '_', text)
        # Убираем начальные и конечные пробелы/точки
        safe = safe.strip('. ')
        # Ограничиваем длину
        return safe[:50]
    
    def get_summary(self):
        """Получение сводки по всем листам"""
        if not self.sheet_data:
            return {"error": "Данные не загружены"}
        
        summary = {
            "file": str(self.file_path.name),
            "type": self.data_type,
            "total_sheets": len(self.sheet_data),
            "sheets": {}
        }
        
        for sheet_name, df in self.sheet_data.items():
            sheet_info = {
                "rows": len(df),
                "columns": len(df.columns),
                "transformations": len(self.transformations_log.get(sheet_name, [])),
                "columns_list": list(df.columns)
            }
            
            # Информация о типах данных
            dtypes = df.dtypes.value_counts().to_dict()
            sheet_info["dtypes"] = {str(k): int(v) for k, v in dtypes.items()}
            
            # Ищем колонки с датами
            date_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)
            
            sheet_info["date_columns"] = date_cols
            
            summary["sheets"][sheet_name] = sheet_info
        
        return summary
    
    def print_summary(self):
        """Вывод сводки в консоль"""
        if not self.sheet_data:
            print("Данные не загружены")
            return
        
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("СВОДКА ПО ФАЙЛУ".center(70))
        print("=" * 70)
        
        print(f"Файл: {summary['file']}")
        print(f"Тип: {summary['type'].upper() if summary['type'] else 'Неизвестно'}")
        print(f"Листов: {summary['total_sheets']}")
        print("-" * 70)
        
        for sheet_name, sheet_info in summary['sheets'].items():
            print(f"\n{sheet_name}:")
            print(f"  Строк: {sheet_info['rows']:,}")
            print(f"  Колонок: {sheet_info['columns']}")
            print(f"  Преобразований: {sheet_info['transformations']}")
            
            # Показываем колонки с датами
            if sheet_info['date_columns']:
                print(f"  Колонки с датами: {', '.join(sheet_info['date_columns'])}")
            
            # Показываем типы данных
            if sheet_info['dtypes']:
                print(f"  Типы данных:")
                for dtype, count in sheet_info['dtypes'].items():
                    print(f"    • {dtype}: {count} колонок")
        
        print("\n" + "=" * 70)


# Интерактивный режим с полным функционалом
def interactive_mode():
    """Интерактивный режим работы"""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ОЧИСТКА МНОГОЛИСТОВЫХ EXCEL ФАЙЛОВ                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Выбор файла
    while True:
        file_path = input("\nВведите путь к Excel файлу: ").strip()
        
        if not file_path:
            print("Отмена")
            return
        
        if not os.path.exists(file_path):
            print(f"✗ Файл не найден: {file_path}")
            continue
        
        if not file_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            print("⚠ Файл не является Excel файлом (.xlsx, .xls, .xlsm)")
            continue
        
        break
    
    # Создаем очиститель
    cleaner = MultiSheetDataCleaner(file_path)
    
    # Загружаем данные
    print("\nЗагрузка данных...")
    sheets = cleaner.load_all_sheets()
    
    if not sheets:
        return
    
    # Меню
    while True:
        print("\n" + "-" * 60)
        print("ГЛАВНОЕ МЕНЮ")
        print("-" * 60)
        print("1. Очистить все листы (все операции)")
        print("2. Очистить с выбором операций")
        print("3. Собрать колонки даты в одну (год, месяц, день → дата)")
        print("4. Извлечь части даты (год, месяц, день и т.д.)")
        print("5. Показать сводку по листам")
        print("6. Сохранить как Excel (все листы в один файл)")
        print("7. Сохранить как CSV (каждый лист в отдельный файл)")
        print("8. Работа с отдельным листом")
        print("9. Выход")
        
        choice = input("\nВаш выбор (1-9): ").strip()
        
        if choice == '1':
            # Очистить все листы
            print("\nОчистка всех листов...")
            cleaner.clean_all_sheets()
            print("✓ Все листы очищены")
            
        elif choice == '2':
            # Очистить с выбором операций
            print("\n" + "-" * 60)
            print("ВЫБОР ОПЕРАЦИЙ")
            print("-" * 60)
            print("Доступные операции:")
            print("  1. Очистить имена колонок")
            print("  2. Собрать колонки даты (год, месяц, день)")
            print("  3. Преобразовать координаты")
            print("  4. Обработать пропущенные значения")
            print("  5. Определить типы данных")
            print("  6. Преобразовать форматы дат")
            print("  7. Все операции")
            
            op_choice = input("\nВыберите операции (через запятую, например 1,2,3): ").strip()
            
            operations = []
            op_map = {
                '1': 'clean_names',
                '2': 'merge_dates',
                '3': 'parse_coords',
                '4': 'handle_missing',
                '5': 'detect_types',
                '6': 'convert_dates',
                '7': 'all'
            }
            
            if op_choice == '7':
                operations = ['clean_names', 'merge_dates', 'parse_coords', 
                            'handle_missing', 'detect_types', 'convert_dates']
            else:
                for num in op_choice.split(','):
                    num = num.strip()
                    if num in op_map and num != '7':
                        operations.append(op_map[num])
            
            if operations:
                print(f"\nВыполнение операций: {', '.join(operations)}")
                cleaner.clean_all_sheets(operations)
            else:
                print("Не выбрано ни одной операции")
            
        elif choice == '3':
            # Собрать колонки даты
            print("\n" + "-" * 60)
            print("СОБРАТЬ КОЛОНКИ ДАТЫ В ОДНУ")
            print("-" * 60)
            
            # Спросим, нужно ли сохранять исходные колонки
            keep_original = input("Сохранить исходные колонки даты? (y/n): ").lower().strip() == 'y'
            
            for sheet_name in cleaner.sheet_data.keys():
                df = cleaner.sheet_data[sheet_name]
                
                # Находим колонки с датами
                date_cols = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(word in col_lower for word in ['год', 'месяц', 'день', 'date', 'year', 'month', 'day']):
                        date_cols.append(col)
                
                if len(date_cols) >= 2:
                    print(f"\nЛист '{sheet_name}':")
                    print(f"  Найдены колонки: {', '.join(date_cols)}")
                    
                    # Создаем новую колонку даты
                    cleaner._merge_date_columns_simple(df, sheet_name, keep_original)
                else:
                    print(f"\nЛист '{sheet_name}':")
                    print(f"  Не найдены колонки для объединения даты")
            
            print("\n✓ Колонки даты обработаны")
            
        elif choice == '4':
            # Преобразовать форматы дат
            print("\nПреобразование форматов дат...")
            for sheet_name in cleaner.sheet_data.keys():
                df = cleaner.sheet_data[sheet_name]
                cleaner._convert_date_formats(df, sheet_name)
            print("✓ Форматы дат преобразованы")
            
        elif choice == '5':
            # Извлечь части даты
            print("\nИзвлечение частей даты...")
            for sheet_name in cleaner.sheet_data.keys():
                cleaner.extract_date_parts_in_sheet(sheet_name)
            print("✓ Части даты извлечены")
            
        elif choice == '6':
            # Сохранить как Excel
            print("\n" + "-" * 60)
            print("СОХРАНЕНИЕ В EXCEL")
            print("-" * 60)
            
            print("Варианты сохранения:")
            print("  1. Рядом с исходным файлом")
            print("  2. На Рабочем столе")
            print("  3. Указать свой путь")
            
            save_choice = input("\nВаш выбор (1-3): ").strip()
            
            if save_choice == '1':
                output_path = None  # Автоматически
            elif save_choice == '2':
                desktop = Path.home() / "Desktop"
                default_name = f"{Path(file_path).stem}_очищенный.xlsx"
                output_path = desktop / default_name
            elif save_choice == '3':
                custom_path = input("Введите полный путь к файлу: ").strip()
                if custom_path:
                    output_path = Path(custom_path)
                else:
                    print("Использую путь по умолчанию")
                    output_path = None
            else:
                output_path = None
            
            saved_path = cleaner.save_to_excel(output_path)
            
            if saved_path:
                print(f"\n✓ Файл сохранен: {saved_path}")
                
                # Предлагаем открыть
                if os.name == 'nt':  # Windows
                    open_file = input("Открыть файл? (y/n): ").lower().strip()
                    if open_file == 'y':
                        try:
                            os.startfile(saved_path)
                        except:
                            print("Не удалось открыть файл")
            
        elif choice == '7':
            # Сохранить как CSV
            print("\n" + "-" * 60)
            print("СОХРАНЕНИЕ В CSV")
            print("-" * 60)
            
            print("Выберите директорию для сохранения:")
            print("  1. Создать папку рядом с исходным файлом")
            print("  2. На Рабочем столе")
            print("  3. Указать свою директорию")
            
            csv_choice = input("\nВаш выбор (1-3): ").strip()
            
            if csv_choice == '1':
                output_dir = None  # Автоматически
            elif csv_choice == '2':
                desktop = Path.home() / "Desktop"
                dir_name = f"{Path(file_path).stem}_csv"
                output_dir = desktop / dir_name
            elif csv_choice == '3':
                custom_dir = input("Введите путь к директории: ").strip()
                if custom_dir:
                    output_dir = Path(custom_dir)
                else:
                    print("Использую путь по умолчанию")
                    output_dir = None
            else:
                output_dir = None
            
            saved_files = cleaner.save_to_csv(output_dir)
            
            if saved_files:
                print(f"\n✓ Сохранено {len(saved_files)} CSV файлов")
                
                # Открыть директорию
                if saved_files and os.name == 'nt':
                    open_dir = input("Открыть папку с файлами? (y/n): ").lower().strip()
                    if open_dir == 'y':
                        try:
                            os.startfile(saved_files[0].parent)
                        except:
                            pass
        
        elif choice == '8':
            # Работа с отдельным листом
            print("\n" + "-" * 60)
            print("РАБОТА С ОТДЕЛЬНЫМ ЛИСТОМ")
            print("-" * 60)
            
            sheet_names = list(cleaner.sheet_data.keys())
            print(f"Доступные листы ({len(sheet_names)}):")
            
            for i, name in enumerate(sheet_names, 1):
                df = cleaner.sheet_data[name]
                print(f"  {i}. {name} ({len(df)} строк, {len(df.columns)} колонок)")
            
            sheet_choice = input("\nВыберите номер листа (или 0 для отмены): ").strip()
            
            if sheet_choice.isdigit():
                idx = int(sheet_choice) - 1
                if 0 <= idx < len(sheet_names):
                    sheet_name = sheet_names[idx]
                    _work_with_sheet(cleaner, sheet_name)
            
        elif choice == '9':
            # Выход
            confirm = input("\nВы уверены, что хотите выйти? (y/n): ").lower().strip()
            if confirm == 'y':
                print("\nВыход из программы")
                break
        
        else:
            print("Неверный выбор. Попробуйте еще раз.")


def _work_with_sheet(cleaner, sheet_name):
    """Работа с отдельным листом"""
    df = cleaner.sheet_data[sheet_name]
    
    while True:
        print(f"\n" + "-" * 60)
        print(f"ЛИСТ: {sheet_name}")
        print("-" * 60)
        print(f"Строк: {len(df):,}, Колонок: {len(df.columns)}")
        print("\nОперации:")
        print("  1. Показать информацию о колонках")
        print("  2. Преобразовать даты")
        print("  3. Извлечь части даты")
        print("  4. Преобразовать координаты")
        print("  5. Обработать пропущенные значения")
        print("  6. Показать первые строки")
        print("  7. Назад")
        
        choice = input("\nВаш выбор (1-7): ").strip()
        
        if choice == '1':
            # Показать информацию о колонках
            print(f"\nКолонки листа '{sheet_name}':")
            for i, col in enumerate(df.columns, 1):
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                unique = df[col].nunique()
                print(f"  {i:2d}. {col:25} {dtype:12} {non_null:6} непустых, {unique:6} уникальных")
            
            input("\nНажмите Enter для продолжения...")
            
        elif choice == '2':
            # Преобразовать даты
            print("\nПреобразование дат...")
            cleaner.convert_dates_in_sheet(sheet_name)
            df = cleaner.sheet_data[sheet_name]
            print("✓ Даты преобразованы")
            
        elif choice == '3':
            # Извлечь части даты
            print("\nИзвлечение частей даты...")
            cleaner.extract_date_parts_in_sheet(sheet_name)
            df = cleaner.sheet_data[sheet_name]
            print("✓ Части даты извлечены")
            
        elif choice == '4':
            # Преобразовать координаты
            print("\nПреобразование координат...")
            df = cleaner._parse_coordinates(df, sheet_name)
            cleaner.sheet_data[sheet_name] = df
            print("✓ Координаты преобразованы")
            
        elif choice == '5':
            # Обработать пропущенные значения
            print("\nОбработка пропущенных значений...")
            df = cleaner._handle_missing_values(df, sheet_name)
            cleaner.sheet_data[sheet_name] = df
            print("✓ Пропущенные значения обработаны")
            
        elif choice == '6':
            # Показать первые строки
            print(f"\nПервые 5 строк листа '{sheet_name}':")
            print(df.head().to_string())
            
            show_more = input("\nПоказать следующие 5 строк? (y/n): ").lower().strip()
            if show_more == 'y':
                print("\nСтроки 6-10:")
                print(df.iloc[5:10].to_string())
            
            input("\nНажмите Enter для продолжения...")
            
        elif choice == '7':
            # Назад
            break
        
        else:
            print("Неверный выбор. Попробуйте еще раз.")


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Очистка многолистовых Excel файлов с поддержкой дат',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s файл.xlsx                    # Обработать и сохранить рядом
  %(prog)s файл.xlsx --output result.xlsx  # Сохранить с указанным именем
  %(prog)s --interactive                # Запустить интерактивный режим
  %(prog)s файл.xlsx --dates-only       # Только преобразовать даты
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Путь к Excel файлу')
    parser.add_argument('--output', '-o', help='Путь для сохранения результата')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Интерактивный режим')
    parser.add_argument('--csv', '-c', action='store_true',
                       help='Сохранить как CSV файлы')
    parser.add_argument('--dates-only', action='store_true',
                       help='Только преобразовать даты')
    parser.add_argument('--extract-dates', action='store_true',
                       help='Извлечь части даты (год, месяц, день)')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Интерактивный режим
        interactive_mode()
    
    elif args.input_file:
        # Обработка файла
        cleaner = MultiSheetDataCleaner(args.input_file)
        cleaner.load_all_sheets()
        
        if args.dates_only:
            # Только преобразование дат
            for sheet_name in cleaner.sheet_data.keys():
                cleaner.convert_dates_in_sheet(sheet_name)
        elif args.extract_dates:
            # Извлечение частей даты
            for sheet_name in cleaner.sheet_data.keys():
                cleaner.extract_date_parts_in_sheet(sheet_name)
        else:
            # Полная очистка
            cleaner.clean_all_sheets()
        
        if args.csv:
            # Сохранить как CSV
            cleaner.save_to_csv(args.output)
        else:
            # Сохранить как Excel
            cleaner.save_to_excel(args.output)
    
    else:
        # Если аргументов нет, запускаем интерактивный режим
        print("Запуск в интерактивном режиме...")
        interactive_mode()


if __name__ == "__main__":
    main()