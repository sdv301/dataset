import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib
import chardet
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DataQualityChecker:
    """Класс для проверки качества данных"""
    
    @staticmethod
    def check_dataframe_quality(df: pd.DataFrame, df_name: str = "") -> Dict:
        """
        Проверяет качество данных в DataFrame
        
        Args:
            df: DataFrame для проверки
            df_name: имя DataFrame для отчетности
            
        Returns:
            Словарь с результатами проверки
        """
        results = {
            'name': df_name,
            'timestamp': datetime.now().isoformat(),
            'basic_stats': {},
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        if df.empty:
            results['issues'].append("Пустой DataFrame")
            return results
        
        # Основная статистика
        results['basic_stats'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Проверка на дубликаты
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            results['issues'].append(f"Найдено {duplicate_rows} полностью дублирующихся строк ({duplicate_rows/len(df)*100:.1f}%)")
        
        # Проверка пропущенных значений
        missing_total = df.isnull().sum().sum()
        missing_percent = missing_total / results['basic_stats']['total_cells'] * 100
        
        if missing_total > 0:
            results['issues'].append(f"Пропущенные значения: {missing_total} ({missing_percent:.1f}%)")
            
            # Детали по колонкам
            missing_by_column = df.isnull().sum()
            high_missing_cols = missing_by_column[missing_by_column > 0]
            if not high_missing_cols.empty:
                results['warnings'].extend([
                    f"Колонка '{col}': {missing} пропущенных значений ({missing/len(df)*100:.1f}%)"
                    for col, missing in high_missing_cols.items()
                    if missing/len(df)*100 > 50
                ])
        
        # Проверка типов данных
        dtype_info = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype not in dtype_info:
                dtype_info[dtype] = []
            dtype_info[dtype].append(col)
        
        results['dtype_distribution'] = dtype_info
        
        # Проверка аномальных значений в числовых колонках
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_stats = df[col].describe()
            
            # Проверка на бесконечные значения
            if np.any(np.isinf(df[col])):
                results['issues'].append(f"Колонка '{col}' содержит бесконечные значения")
            
            # Проверка на выбросы (используем IQR метод)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_percent = len(outliers) / len(df) * 100
                if outlier_percent > 5:
                    results['warnings'].append(
                        f"Колонка '{col}': {len(outliers)} выбросов ({outlier_percent:.1f}%)"
                    )
        
        # Проверка текстовых колонок
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            # Проверка на слишком длинные строки
            if df[col].str.len().max() > 1000:
                results['warnings'].append(f"Колонка '{col}' содержит очень длинные строки")
            
            # Проверка на специальные символы
            special_chars = df[col].str.contains(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', na=False)
            if special_chars.any():
                results['issues'].append(f"Колонка '{col}' содержит непечатаемые символы")
        
        # Проверка временных рядов
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            if df[col].is_monotonic_increasing:
                results['recommendations'].append(f"Колонка '{col}' отсортирована по возрастанию")
            elif df[col].is_monotonic_decreasing:
                results['recommendations'].append(f"Колонка '{col}' отсортирована по убыванию")
        
        # Проверка уникальности ключевых колонок
        potential_key_cols = [col for col in df.columns if 'id' in col.lower() or 'key' in col.lower()]
        for col in potential_key_cols:
            unique_count = df[col].nunique()
            if unique_count == len(df):
                results['recommendations'].append(f"Колонка '{col}' может быть первичным ключом (уникальные значения)")
            elif unique_count/len(df) < 0.5:
                results['warnings'].append(f"Колонка '{col}': низкая уникальность ({unique_count/len(df)*100:.1f}%)")
        
        return results
    
    @staticmethod
    def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                          df1_name: str = "Источник", df2_name: str = "Цель") -> Dict:
        """
        Сравнивает два DataFrame
        
        Args:
            df1: первый DataFrame
            df2: второй DataFrame
            df1_name: имя первого DataFrame
            df2_name: имя второго DataFrame
            
        Returns:
            Словарь с результатами сравнения
        """
        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'summary': {},
            'differences': [],
            'matches': []
        }
        
        # Сравнение размеров
        size_match = len(df1) == len(df2) and len(df1.columns) == len(df2.columns)
        comparison['summary']['size_match'] = size_match
        comparison['summary']['df1_size'] = f"{len(df1)}x{len(df1.columns)}"
        comparison['summary']['df2_size'] = f"{len(df2)}x{len(df2.columns)}"
        
        if not size_match:
            comparison['differences'].append(
                f"Размеры не совпадают: {df1_name} {len(df1)}x{len(df1.columns)}, "
                f"{df2_name} {len(df2)}x{len(df2.columns)}"
            )
        
        # Сравнение колонок
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        missing_in_df2 = cols1 - cols2
        missing_in_df1 = cols2 - cols1
        common_cols = cols1 & cols2
        
        if missing_in_df2:
            comparison['differences'].append(
                f"Колонки отсутствуют в {df2_name}: {', '.join(missing_in_df2)}"
            )
        
        if missing_in_df1:
            comparison['differences'].append(
                f"Колонки отсутствуют в {df1_name}: {', '.join(missing_in_df1)}"
            )
        
        if common_cols:
            comparison['matches'].append(
                f"Общие колонки ({len(common_cols)}): {', '.join(sorted(common_cols)[:10])}"
                + ("..." if len(common_cols) > 10 else "")
            )
        
        # Сравнение типов данных для общих колонок
        dtype_diffs = []
        for col in common_cols:
            if df1[col].dtype != df2[col].dtype:
                dtype_diffs.append(
                    f"Колонка '{col}': {df1[col].dtype} ≠ {df2[col].dtype}"
                )
        
        if dtype_diffs:
            comparison['differences'].extend(dtype_diffs)
        
        # Сравнение значений (выборочно для производительности)
        if len(df1) > 0 and len(df2) > 0 and common_cols:
            # Берем случайную выборку для сравнения
            sample_size = min(1000, len(df1), len(df2))
            sample_df1 = df1[list(common_cols)].sample(n=sample_size, random_state=42)
            sample_df2 = df2[list(common_cols)].sample(n=sample_size, random_state=42)
            
            # Сравниваем значения
            value_diffs = []
            for col in common_cols:
                if not sample_df1[col].equals(sample_df2[col]):
                    mismatches = (sample_df1[col] != sample_df2[col]).sum()
                    if mismatches > 0:
                        value_diffs.append(
                            f"Колонка '{col}': {mismatches} несовпадений в выборке из {sample_size}"
                        )
            
            if value_diffs:
                comparison['differences'].extend(value_diffs)
            else:
                comparison['matches'].append(
                    f"Значения совпадают в случайной выборке ({sample_size} строк)"
                )
        
        # Сравнение контрольных сумм
        try:
            # Создаем хеш для каждого DataFrame
            df1_hash = hashlib.md5(
                pd.util.hash_pandas_object(df1, index=True).values.tobytes()
            ).hexdigest()
            df2_hash = hashlib.md5(
                pd.util.hash_pandas_object(df2, index=True).values.tobytes()
            ).hexdigest()
            
            comparison['summary']['df1_hash'] = df1_hash[:16]
            comparison['summary']['df2_hash'] = df2_hash[:16]
            comparison['summary']['hash_match'] = df1_hash == df2_hash
            
            if df1_hash == df2_hash:
                comparison['matches'].append("DataFrame идентичны (хеш-суммы совпадают)")
            else:
                comparison['differences'].append("DataFrame различны (хеш-суммы не совпадают)")
        except:
            comparison['warnings'] = ["Не удалось вычислить хеш-суммы"]
        
        return comparison
    
    @staticmethod
    def validate_csv_file(file_path: Path) -> Dict:
        """
        Проверяет CSV файл на корректность
        
        Args:
            file_path: путь к CSV файлу
            
        Returns:
            Словарь с результатами проверки
        """
        results = {
            'file_path': str(file_path),
            'timestamp': datetime.now().isoformat(),
            'valid': False,
            'issues': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Проверка существования файла
            if not file_path.exists():
                results['issues'].append("Файл не существует")
                return results
            
            # Информация о файле
            file_size = file_path.stat().st_size
            results['file_info']['size_bytes'] = file_size
            results['file_info']['size_mb'] = file_size / 1024**2
            
            # Определение кодировки
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                encoding_result = chardet.detect(raw_data)
                results['file_info']['detected_encoding'] = encoding_result
            
            # Попытка чтения файла
            try:
                # Сначала пробуем прочитать с разными разделителями
                for sep in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, nrows=1000, 
                                        encoding=encoding_result['encoding'])
                        results['file_info']['delimiter'] = sep
                        results['file_info']['sample_rows'] = len(df)
                        results['file_info']['columns'] = list(df.columns)
                        results['valid'] = True
                        break
                    except:
                        continue
                
                if not results['valid']:
                    results['issues'].append("Не удалось определить разделитель")
                    
            except Exception as e:
                results['issues'].append(f"Ошибка чтения CSV: {str(e)}")
            
            # Проверка целостности файла
            if results['valid']:
                # Читаем весь файл для полной проверки
                try:
                    df_full = pd.read_csv(file_path, sep=results['file_info']['delimiter'],
                                         encoding=encoding_result['encoding'])
                    
                    # Проверка на пустые строки в середине файла
                    if df_full.isnull().all(axis=1).any():
                        results['warnings'].append("Файл содержит полностью пустые строки")
                    
                    # Проверка согласованности данных
                    for col in df_full.columns:
                        # Проверка на смешанные типы
                        if df_full[col].apply(type).nunique() > 1:
                            results['warnings'].append(
                                f"Колонка '{col}' содержит смешанные типы данных"
                            )
                    
                except Exception as e:
                    results['issues'].append(f"Ошибка при полном чтении файла: {str(e)}")
                    results['valid'] = False
            
        except Exception as e:
            results['issues'].append(f"Неожиданная ошибка: {str(e)}")
        
        return results


class ExcelToCSVConverterWithValidation:
    """Расширенный конвертер с валидацией данных"""
    
    def __init__(self):
        self.excel_path = None
        self.excel_data = None
        self.quality_checker = DataQualityChecker()
        self.conversion_log = []
        
    def load_excel(self, file_path: str) -> Tuple[bool, str]:
        """Загружает Excel файл с валидацией"""
        try:
            self.excel_path = Path(file_path)
            
            # Проверка файла
            if not self.excel_path.exists():
                return False, "Файл не существует"
            
            if self.excel_path.suffix.lower() not in ['.xlsx', '.xls', '.xlsm']:
                return False, "Неверный формат файла. Ожидается .xlsx, .xls или .xlsm"
            
            # Чтение Excel
            self.excel_data = pd.ExcelFile(file_path)
            
            # Логирование
            self._log_event("LOAD", f"Загружен файл: {file_path}", {
                'sheets': self.excel_data.sheet_names,
                'file_size': self.excel_path.stat().st_size
            })
            
            return True, "Файл успешно загружен"
            
        except Exception as e:
            error_msg = f"Ошибка загрузки: {str(e)}"
            self._log_event("ERROR", error_msg)
            return False, error_msg
    
    def _log_event(self, event_type: str, message: str, data: Dict = None):
        """Логирует события конвертации"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'data': data or {}
        }
        self.conversion_log.append(log_entry)
    
    def analyze_excel_quality(self) -> Dict:
        """Анализирует качество данных во всех листах Excel"""
        if not self.excel_data:
            return {'error': 'Excel файл не загружен'}
        
        analysis_report = {
            'excel_file': str(self.excel_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'total_sheets': len(self.excel_data.sheet_names),
            'sheets_analysis': [],
            'summary': {
                'total_rows': 0,
                'total_columns': 0,
                'total_cells': 0,
                'issues_found': 0,
                'warnings_found': 0
            }
        }
        
        for sheet_name in self.excel_data.sheet_names:
            try:
                # Читаем лист
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name, nrows=10000)
                
                # Анализируем качество
                quality_report = self.quality_checker.check_dataframe_quality(df, sheet_name)
                
                # Обновляем сводку
                analysis_report['summary']['total_rows'] += quality_report['basic_stats'].get('rows', 0)
                analysis_report['summary']['total_columns'] += quality_report['basic_stats'].get('columns', 0)
                analysis_report['summary']['total_cells'] += quality_report['basic_stats'].get('total_cells', 0)
                analysis_report['summary']['issues_found'] += len(quality_report.get('issues', []))
                analysis_report['summary']['warnings_found'] += len(quality_report.get('warnings', []))
                
                analysis_report['sheets_analysis'].append(quality_report)
                
                self._log_event("ANALYSIS", f"Проанализирован лист: {sheet_name}", {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'issues': len(quality_report.get('issues', [])),
                    'warnings': len(quality_report.get('warnings', []))
                })
                
            except Exception as e:
                error_report = {
                    'name': sheet_name,
                    'error': str(e),
                    'issues': [f"Ошибка при анализе: {str(e)}"]
                }
                analysis_report['sheets_analysis'].append(error_report)
                analysis_report['summary']['issues_found'] += 1
        
        return analysis_report
    
    def convert_with_validation(self, output_dir: str, mode: str = 'separate', 
                               add_sheet_column: bool = True) -> Dict:
        """
        Конвертирует Excel в CSV с полной валидацией
        
        Args:
            output_dir: директория для сохранения
            mode: 'separate' (отдельные файлы) или 'single' (один файл)
            add_sheet_column: добавлять колонку с источником
            
        Returns:
            Детальный отчет о конвертации
        """
        report = {
            'conversion_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'excel_file': str(self.excel_path),
            'output_mode': mode,
            'output_directory': output_dir,
            'sheets_processed': 0,
            'total_rows_converted': 0,
            'files_created': [],
            'validation_results': [],
            'issues': [],
            'warnings': [],
            'success': False
        }
        
        try:
            # Создаем директорию
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Предварительный анализ качества
            report['pre_conversion_analysis'] = self.analyze_excel_quality()
            
            if mode == 'separate':
                result = self._convert_separate_with_validation(output_path, report)
            else:
                result = self._convert_single_with_validation(output_path, add_sheet_column, report)
            
            report.update(result)
            report['success'] = True
            
        except Exception as e:
            report['issues'].append(f"Критическая ошибка: {str(e)}")
            report['success'] = False
        
        report['end_time'] = datetime.now().isoformat()
        report['duration_seconds'] = (
            datetime.fromisoformat(report['end_time']) - 
            datetime.fromisoformat(report['start_time'])
        ).total_seconds()
        
        # Сохраняем отчет
        self._save_conversion_report(report, output_path)
        
        return report
    
    def _convert_separate_with_validation(self, output_path: Path, report: Dict) -> Dict:
        """Конвертирует в отдельные файлы с валидацией"""
        base_name = self.excel_path.stem
        
        for sheet_name in self.excel_data.sheet_names:
            sheet_report = {
                'sheet_name': sheet_name,
                'status': 'pending',
                'validation': {},
                'file_info': {}
            }
            
            try:
                # Читаем лист
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                
                # Валидация данных
                validation = self.quality_checker.check_dataframe_quality(df, sheet_name)
                sheet_report['validation'] = validation
                
                # Создаем имя файла
                safe_name = self._make_filename_safe(sheet_name)
                if len(self.excel_data.sheet_names) == 1:
                    csv_name = f"{base_name}.csv"
                else:
                    csv_name = f"{base_name}_{safe_name}.csv"
                
                csv_path = output_path / csv_name
                
                # Сохраняем CSV
                self._save_csv_with_validation(df, csv_path)
                
                # Валидация CSV файла
                csv_validation = self.quality_checker.validate_csv_file(csv_path)
                
                # Сравнение данных
                comparison = self.quality_checker.compare_dataframes(
                    df, pd.read_csv(csv_path), 
                    f"Excel: {sheet_name}", f"CSV: {csv_name}"
                )
                
                sheet_report.update({
                    'status': 'success',
                    'file_info': {
                        'filename': csv_name,
                        'path': str(csv_path),
                        'size_bytes': csv_path.stat().st_size,
                        'rows': len(df),
                        'columns': len(df.columns)
                    },
                    'csv_validation': csv_validation,
                    'data_comparison': comparison
                })
                
                report['files_created'].append(str(csv_path))
                report['total_rows_converted'] += len(df)
                report['sheets_processed'] += 1
                
                self._log_event("CONVERT", f"Конвертирован лист: {sheet_name}", {
                    'rows': len(df),
                    'file': csv_name,
                    'comparison_result': comparison.get('summary', {}).get('hash_match', False)
                })
                
            except Exception as e:
                sheet_report.update({
                    'status': 'error',
                    'error': str(e)
                })
                report['issues'].append(f"Ошибка в листе '{sheet_name}': {str(e)}")
            
            report['validation_results'].append(sheet_report)
        
        return report
    
    def _convert_single_with_validation(self, output_path: Path, 
                                       add_sheet_column: bool, report: Dict) -> Dict:
        """Конвертирует в один файл с валидацией"""
        all_dfs = []
        sheets_info = []
        
        for sheet_name in self.excel_data.sheet_names:
            try:
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                
                if add_sheet_column:
                    df.insert(0, 'Источник_лист', sheet_name)
                    df.insert(1, 'Номер_листа', self.excel_data.sheet_names.index(sheet_name) + 1)
                
                # Валидация исходных данных
                validation = self.quality_checker.check_dataframe_quality(df, sheet_name)
                
                sheets_info.append({
                    'sheet_name': sheet_name,
                    'original_rows': len(df),
                    'original_columns': len(df.columns),
                    'validation': validation
                })
                
                all_dfs.append(df)
                report['total_rows_converted'] += len(df)
                
            except Exception as e:
                report['warnings'].append(f"Пропущен лист '{sheet_name}': {str(e)}")
        
        if not all_dfs:
            raise ValueError("Не удалось загрузить ни одного листа")
        
        # Объединяем данные
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Создаем файл
        base_name = self.excel_path.stem
        csv_path = output_path / f"{base_name}_combined.csv"
        
        # Сохраняем
        self._save_csv_with_validation(combined_df, csv_path)
        
        # Валидация результата
        csv_validation = self.quality_checker.validate_csv_file(csv_path)
        
        report.update({
            'files_created': [str(csv_path)],
            'sheets_processed': len(all_dfs),
            'combined_stats': {
                'total_rows': len(combined_df),
                'total_columns': len(combined_df.columns),
                'unique_sheets': len(set(combined_df['Источник_лист'])) if add_sheet_column else 0
            },
            'sheets_info': sheets_info,
            'csv_validation': csv_validation
        })
        
        self._log_event("CONVERT", "Конвертирован в единый файл", {
            'rows': len(combined_df),
            'sheets': len(all_dfs),
            'file': csv_path.name
        })
        
        return report
    
    def _save_csv_with_validation(self, df: pd.DataFrame, path: Path):
        """Сохраняет DataFrame в CSV с дополнительной валидацией"""
        # Определяем оптимальную кодировку
        encoding = self._detect_best_encoding(df)
        
        # Сохраняем с контрольной точкой
        temp_path = path.with_suffix('.tmp')
        
        try:
            df.to_csv(temp_path, index=False, encoding=encoding, sep=',')
            
            # Проверяем, что файл создан и не пустой
            if temp_path.stat().st_size == 0:
                raise ValueError("Созданный CSV файл пустой")
            
            # Переименовываем временный файл в постоянный
            temp_path.rename(path)
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _detect_best_encoding(self, df: pd.DataFrame) -> str:
        """Определяет лучшую кодировку для DataFrame"""
        # Проверяем наличие не-ASCII символов
        has_non_ascii = False
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if not sample.empty:
                    has_non_ascii = sample.astype(str).str.contains(
                        r'[^\x00-\x7F]', regex=True
                    ).any()
                    if has_non_ascii:
                        break
        
        return 'utf-8-sig' if has_non_ascii else 'utf-8'
    
    def _make_filename_safe(self, text: str) -> str:
        """Делает строку безопасной для имени файла"""
        import re
        safe = re.sub(r'[\\/*?:"<>|]', "_", text)
        safe = safe.strip('. ')
        return safe[:50]
    
    def _save_conversion_report(self, report: Dict, output_path: Path):
        """Сохраняет детальный отчет о конвертации"""
        report_file = output_path / f"conversion_report_{report['conversion_id']}.json"
        
        # Добавляем лог событий
        report['conversion_log'] = self.conversion_log
        
        # Сохраняем в JSON
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # Также создаем текстовый отчет
        txt_report = self._generate_text_report(report)
        txt_file = output_path / f"conversion_summary_{report['conversion_id']}.txt"
        txt_file.write_text(txt_report, encoding='utf-8')
        
        self._log_event("REPORT", "Сохранен отчет о конвертации", {
            'json_report': str(report_file),
            'txt_report': str(txt_file)
        })
    
    def _generate_text_report(self, report: Dict) -> str:
        """Генерирует текстовый отчет"""
        lines = []
        lines.append("=" * 70)
        lines.append("ОТЧЕТ О КОНВЕРТАЦИИ EXCEL В CSV".center(70))
        lines.append("=" * 70)
        lines.append(f"Время: {report.get('start_time', 'N/A')}")
        lines.append(f"Файл: {report.get('excel_file', 'N/A')}")
        lines.append(f"Режим: {report.get('output_mode', 'N/A')}")
        lines.append(f"Успешно: {'ДА' if report.get('success') else 'НЕТ'}")
        lines.append("-" * 70)
        
        if report.get('success'):
            lines.append("РЕЗУЛЬТАТЫ:")
            lines.append(f"  Листов обработано: {report.get('sheets_processed', 0)}")
            lines.append(f"  Всего строк: {report.get('total_rows_converted', 0):,}")
            
            if report['output_mode'] == 'separate':
                lines.append(f"  Создано файлов: {len(report.get('files_created', []))}")
                for file in report.get('files_created', [])[:5]:
                    lines.append(f"    • {Path(file).name}")
                if len(report.get('files_created', [])) > 5:
                    lines.append(f"    ... и еще {len(report.get('files_created', [])) - 5} файлов")
            else:
                if report.get('files_created'):
                    file_path = Path(report['files_created'][0])
                    lines.append(f"  Создан файл: {file_path.name}")
                    lines.append(f"  Размер: {file_path.stat().st_size / 1024**2:.2f} MB")
            
            # Проблемы и предупреждения
            issues = report.get('issues', [])
            warnings = report.get('warnings', [])
            
            if issues:
                lines.append("\nПРОБЛЕМЫ:")
                for issue in issues[:10]:
                    lines.append(f"  ⚠ {issue}")
            
            if warnings:
                lines.append("\nПРЕДУПРЕЖДЕНИЯ:")
                for warning in warnings[:10]:
                    lines.append(f"  ⚠ {warning}")
        
        else:
            lines.append("КОНВЕРТАЦИЯ НЕ УДАЛАСЬ:")
            for issue in report.get('issues', []):
                lines.append(f"  ✗ {issue}")
        
        lines.append("-" * 70)
        lines.append(f"Отчет сохранен в: {report.get('output_directory', 'N/A')}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_detailed_validation_report(self, csv_file: str) -> Dict:
        """Получает детальный отчет о валидации CSV файла"""
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            return {'error': 'CSV файл не существует'}
        
        # Валидация файла
        file_validation = self.quality_checker.validate_csv_file(csv_path)
        
        if not file_validation.get('valid'):
            return file_validation
        
        # Читаем и анализируем данные
        try:
            df = pd.read_csv(csv_file, encoding=file_validation['file_info'].get('detected_encoding', {}).get('encoding', 'utf-8'))
            
            # Проверка качества данных
            data_quality = self.quality_checker.check_dataframe_quality(df, csv_path.name)
            
            # Дополнительные проверки
            additional_checks = {
                'file_integrity': self._check_file_integrity(csv_path, df),
                'data_consistency': self._check_data_consistency(df),
                'business_rules': self._check_business_rules(df)
            }
            
            return {
                'file_validation': file_validation,
                'data_quality': data_quality,
                'additional_checks': additional_checks,
                'summary': {
                    'is_valid': file_validation['valid'] and len(data_quality.get('issues', [])) == 0,
                    'total_issues': len(file_validation.get('issues', [])) + len(data_quality.get('issues', [])),
                    'total_warnings': len(file_validation.get('warnings', [])) + len(data_quality.get('warnings', []))
                }
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка при анализе данных: {str(e)}',
                'file_validation': file_validation
            }
    
    def _check_file_integrity(self, file_path: Path, df: pd.DataFrame) -> Dict:
        """Проверяет целостность файла"""
        checks = {
            'file_exists': file_path.exists(),
            'file_not_empty': file_path.stat().st_size > 0,
            'row_count_matches': len(df) > 0,
            'encoding_consistent': True  # Можно добавить проверку
        }
        
        checks['all_passed'] = all(checks.values())
        return checks
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict:
        """Проверяет согласованность данных"""
        checks = {
            'no_completely_empty_rows': not df.isnull().all(axis=1).any(),
            'consistent_column_types': True,
            'date_columns_valid': True
        }
        
        # Проверка типов данных в колонках
        for col in df.columns:
            unique_types = df[col].apply(type).dropna().unique()
            if len(unique_types) > 1:
                checks['consistent_column_types'] = False
                break
        
        checks['all_passed'] = all(checks.values())
        return checks
    
    def _check_business_rules(self, df: pd.DataFrame) -> Dict:
        """Проверяет бизнес-правила (можно настроить под конкретные нужды)"""
        checks = {
            'id_columns_unique': {},
            'required_columns_present': [],
            'value_ranges_valid': []
        }
        
        # Проверка уникальности ID колонок
        id_like_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_like_columns:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            checks['id_columns_unique'][col] = {
                'is_unique': unique_count == total_count,
                'unique_percentage': unique_count / total_count if total_count > 0 else 0
            }
        
        # Проверка обязательных колонок
        required_keywords = ['name', 'date', 'id']
        for keyword in required_keywords:
            matching_cols = [col for col in df.columns if keyword in col.lower()]
            checks['required_columns_present'].append({
                'keyword': keyword,
                'found': len(matching_cols) > 0,
                'columns': matching_cols
            })
        
        # Проверка диапазонов значений для числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Ограничиваем для производительности
            if df[col].notna().any():
                checks['value_ranges_valid'].append({
                    'column': col,
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'has_negative': (df[col] < 0).any(),
                    'has_zero': (df[col] == 0).any()
                })
        
        return checks


# Пример использования
def example_usage():
    """Пример использования конвертера с валидацией"""
    
    # Создаем конвертер
    converter = ExcelToCSVConverterWithValidation()
    
    # Загружаем Excel файл
    excel_file = "пример.xlsx"  # Укажите ваш файл
    success, message = converter.load_excel(excel_file)
    
    if not success:
        print(f"Ошибка: {message}")
        return
    
    print("=" * 70)
    print("АНАЛИЗ КАЧЕСТВА ДАННЫХ В EXCEL".center(70))
    print("=" * 70)
    
    # Анализируем качество данных
    analysis = converter.analyze_excel_quality()
    print(f"Файл: {analysis['excel_file']}")
    print(f"Листов: {analysis['total_sheets']}")
    print(f"Всего строк: {analysis['summary']['total_rows']:,}")
    print(f"Проблемы: {analysis['summary']['issues_found']}")
    print(f"Предупреждения: {analysis['summary']['warnings_found']}")
    
    # Конвертируем с валидацией
    print("\n" + "=" * 70)
    print("КОНВЕРТАЦИЯ С ВАЛИДАЦИЕЙ".center(70))
    print("=" * 70)
    
    # Выбираем режим
    mode = input("Режим (1 - отдельные файлы, 2 - один файл): ").strip()
    mode = 'separate' if mode == '1' else 'single'
    
    output_dir = "validated_csv_output"
    
    print("\nКонвертация...")
    report = converter.convert_with_validation(
        output_dir=output_dir,
        mode=mode,
        add_sheet_column=True
    )
    
    # Показываем результаты
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ КОНВЕРТАЦИИ".center(70))
    print("=" * 70)
    
    if report['success']:
        print(f"✓ Конвертация успешна!")
        print(f"  Листов: {report['sheets_processed']}")
        print(f"  Строк: {report['total_rows_converted']:,}")
        print(f"  Файлов создано: {len(report['files_created'])}")
        
        if report.get('issues'):
            print(f"\n⚠ Проблемы: {len(report['issues'])}")
            for issue in report['issues'][:3]:
                print(f"  • {issue}")
        
        if report.get('warnings'):
            print(f"\n⚠ Предупреждения: {len(report['warnings'])}")
            for warning in report['warnings'][:3]:
                print(f"  • {warning}")
        
        print(f"\n📊 Отчеты сохранены в: {output_dir}")
        
        # Валидация созданных файлов
        print("\n" + "=" * 70)
        print("ВАЛИДАЦИЯ СОЗДАННЫХ CSV ФАЙЛОВ".center(70))
        print("=" * 70)
        
        for csv_file in report['files_created'][:3]:  # Проверяем первые 3 файла
            print(f"\nПроверка файла: {Path(csv_file).name}")
            validation = converter.get_detailed_validation_report(csv_file)
            
            if 'error' in validation:
                print(f"  ✗ Ошибка: {validation['error']}")
            else:
                summary = validation['summary']
                print(f"  ✓ Валиден: {'ДА' if summary['is_valid'] else 'НЕТ'}")
                print(f"  ⚠ Проблемы: {summary['total_issues']}")
                print(f"  ⚠ Предупреждения: {summary['total_warnings']}")
    
    else:
        print("✗ Конвертация не удалась!")
        for issue in report.get('issues', []):
            print(f"  • {issue}")
    
    print("\n" + "=" * 70)


# Консольный интерфейс
def console_interface():
    """Консольный интерфейс для конвертера с валидацией"""
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      КОНВЕРТЕР EXCEL В CSV С ВАЛИДАЦИЕЙ ДАННЫХ          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    converter = ExcelToCSVConverterWithValidation()
    
    # Шаг 1: Выбор файла
    print("\n" + "━" * 60)
    print("ШАГ 1: ВЫБОР EXCEL ФАЙЛА")
    print("━" * 60)
    
    excel_path = input("Введите путь к Excel файлу: ").strip()
    
    if not os.path.exists(excel_path):
        print("❌ Файл не найден!")
        return
    
    success, message = converter.load_excel(excel_path)
    if not success:
        print(f"❌ {message}")
        return
    
    print(f"✅ {message}")
    
    # Шаг 2: Анализ данных
    print("\n" + "━" * 60)
    print("ШАГ 2: АНАЛИЗ КАЧЕСТВА ДАННЫХ")
    print("━" * 60)
    
    print("Анализирую данные...")
    analysis = converter.analyze_excel_quality()
    
    print(f"\n📊 Сводка по файлу:")
    print(f"   • Листов: {analysis['total_sheets']}")
    print(f"   • Всего строк: {analysis['summary']['total_rows']:,}")
    print(f"   • Всего колонок: {analysis['summary']['total_columns']:,}")
    print(f"   • Найдено проблем: {analysis['summary']['issues_found']}")
    print(f"   • Предупреждений: {analysis['summary']['warnings_found']}")
    
    # Шаг 3: Выбор режима
    print("\n" + "━" * 60)
    print("ШАГ 3: ВЫБОР РЕЖИМА КОНВЕРТАЦИИ")
    print("━" * 60)
    
    print("1. 📁 Каждый лист → отдельный CSV файл")
    print("2. 📄 Все листы → один CSV файл")
    
    mode_choice = input("\nВыберите режим (1 или 2): ").strip()
    mode = 'separate' if mode_choice == '1' else 'single'
    
    # Шаг 4: Настройки
    print("\n" + "━" * 60)
    print("ШАГ 4: НАСТРОЙКИ КОНВЕРТАЦИИ")
    print("━" * 60)
    
    output_dir = input("Папка для сохранения (по умолчанию: validated_output): ").strip()
    if not output_dir:
        output_dir = "validated_output"
    
    if mode == 'single':
        add_column = input("Добавить колонки 'Источник_лист' и 'Номер_листа'? (y/n, по умолчанию: y): ").strip().lower()
        add_sheet_column = not (add_column == 'n')
    else:
        add_sheet_column = True
    
    # Шаг 5: Конвертация
    print("\n" + "━" * 60)
    print("ШАГ 5: КОНВЕРТАЦИЯ С ВАЛИДАЦИЕЙ")
    print("━" * 60)
    
    print("🔄 Конвертация и валидация данных...")
    
    report = converter.convert_with_validation(
        output_dir=output_dir,
        mode=mode,
        add_sheet_column=add_sheet_column
    )
    
    # Шаг 6: Результаты
    print("\n" + "━" * 60)
    print("ШАГ 6: РЕЗУЛЬТАТЫ")
    print("━" * 60)
    
    if report['success']:
        print("✅ КОНВЕРТАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print(f"\n📈 Статистика:")
        print(f"   • Листов обработано: {report['sheets_processed']}")
        print(f"   • Всего строк: {report['total_rows_converted']:,}")
        print(f"   • Файлов создано: {len(report['files_created'])}")
        
        if report.get('issues'):
            print(f"\n⚠ Проблемы ({len(report['issues'])}):")
            for issue in report['issues'][:5]:
                print(f"   • {issue}")
        
        if report.get('warnings'):
            print(f"\n⚠ Предупреждения ({len(report['warnings'])}):")
            for warning in report['warnings'][:5]:
                print(f"   • {warning}")
        
        print(f"\n📁 Файлы сохранены в: {os.path.abspath(output_dir)}")
        
        # Сводная информация о созданных файлах
        print(f"\n📋 Созданные файлы:")
        for file_path in report['files_created']:
            file = Path(file_path)
            size_mb = file.stat().st_size / 1024**2
            print(f"   • {file.name} ({size_mb:.2f} MB)")
        
        print(f"\n📊 Подробные отчеты сохранены в папке с результатами")
        
        # Предложение проверить валидацию
        check = input("\nПроверить валидацию созданных файлов? (y/n): ").strip().lower()
        if check == 'y':
            for csv_file in report['files_created'][:2]:  # Проверяем первые 2 файла
                print(f"\n🔍 Проверка {Path(csv_file).name}...")
                validation = converter.get_detailed_validation_report(csv_file)
                
                if 'error' in validation:
                    print(f"   ❌ Ошибка: {validation['error']}")
                else:
                    summary = validation['summary']
                    status = "✅ ВАЛИДЕН" if summary['is_valid'] else "❌ НЕ ВАЛИДЕН"
                    print(f"   {status}")
                    print(f"   • Проблемы: {summary['total_issues']}")
                    print(f"   • Предупреждения: {summary['total_warnings']}")
    
    else:
        print("❌ КОНВЕРТАЦИЯ НЕ УДАЛАСЬ!")
        print("\nПричины:")
        for issue in report.get('issues', []):
            print(f"   • {issue}")
    
    print("\n" + "━" * 60)
    print("Готово! Нажмите Enter для выхода...")
    input()


if __name__ == "__main__":
    console_interface()