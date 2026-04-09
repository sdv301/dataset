# predictor.py
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import warnings
# warnings.filterwarnings('ignore') # Removed global warnings suppression

class TimeSeriesPredictor:
    """Класс для предсказаний временных рядов с гибридным подходом: климатология + XGBoost"""
    
    # Фиксированный набор признаков для согласованности обучения и предсказания
    BASE_FEATURES = [
        'month', 'dayofyear', 'dayofweek',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'year_normalized'
    ]
    LAG_DAYS = [1, 7, 14, 30]
    MA_WINDOWS = [7, 14, 30]
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_names = {}  # Сохраняем имена признаков
        self.predictions = {}
        self.feature_importance = {}
        self.climatology = {}  # Климатологический профиль по дню года
        self.year_range = {}  # Диапазон годов для нормализации
        self.date_formats = ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y.%m.%d', '%m/%d/%Y', '%Y%m%d']
    
    def load_all_excel_sheets(self, file_path):
        """Загрузить все листы Excel и объединить данные"""
        try:
            if str(file_path).endswith(('.xlsx', '.xls')):
                # Читаем все листы
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                
                st.info(f"📊 Найдены листы: {', '.join(sheet_names)}")
                
                all_dfs = []
                
                for sheet in sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                        if not df.empty:
                            # Добавляем информацию о листе
                            df['source_sheet'] = sheet
                            all_dfs.append(df)
                            st.success(f"✅ Лист '{sheet}' загружен ({len(df)} строк)")
                    except Exception as e:
                        st.warning(f"⚠️ Ошибка загрузки листа '{sheet}': {str(e)}")
                
                if not all_dfs:
                    raise ValueError("Не удалось загрузить ни один лист")
                
                # Находим общие колонки
                common_columns = None
                for df in all_dfs:
                    if common_columns is None:
                        common_columns = set(df.columns)
                    else:
                        common_columns = common_columns.intersection(set(df.columns))
                
                # Убираем служебные колонки
                common_columns = [col for col in common_columns if col != 'source_sheet']
                
                st.info(f"📋 Общие колонки: {', '.join(common_columns)}")
                
                # Объединяем по общим колонкам
                combined_dfs = []
                for df in all_dfs:
                    # Берем только общие колонки плюс source_sheet
                    available_cols = [col for col in common_columns if col in df.columns]
                    if available_cols:
                        df_filtered = df[available_cols + ['source_sheet']].copy()
                        combined_dfs.append(df_filtered)
                
                if combined_dfs:
                    combined_df = pd.concat(combined_dfs, ignore_index=True)
                    st.success(f"✅ Объединено {len(combined_dfs)} листов, всего {len(combined_df)} строк")
                    return combined_df
                else:
                    # Если нет общих колонок, возвращаем первый лист
                    return all_dfs[0].drop(columns=['source_sheet']) if 'source_sheet' in all_dfs[0].columns else all_dfs[0]
            
            return None
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки Excel файла: {str(e)}")
            return None
    
    def detect_date_columns(self, df):
        """Обнаружить колонки с датами"""
        if df is None or df.empty:
            return []
        
        date_columns = []
        for col in df.columns:
            # Пропускаем служебные колонки
            if col == 'source_sheet':
                continue
                
            # Проверяем типы данных
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(col)
            else:
                # Пробуем преобразовать в дату
                try:
                    sample_values = df[col].dropna().head(10)
                    if not sample_values.empty:
                        # Пробуем разные форматы
                        converted = pd.to_datetime(df[col], errors='coerce', format='mixed')
                        # Проверяем, что хотя бы 30% значений успешно преобразовались
                        if converted.notna().sum() > len(df) * 0.3:
                            date_columns.append(col)
                except Exception as e:
                    continue
        
        return date_columns
    
    def _build_climatology_profile(self, df, target_column, date_column):
        """Построить климатологический профиль — среднее по дню года за все годы.
        
        Для каждого дня года (1-366) вычисляется:
        - mean, median, std
        - квантили q10, q25, q75, q90
        - min, max, count
        
        Затем профиль сглаживается скользящим окном ±7 дней.
        """
        df_work = df.copy()
        
        # Очищаем аномалии перед построением профиля
        df_work = self._handle_anomalies(df_work, target_column)
        
        if not pd.api.types.is_datetime64_any_dtype(df_work[date_column]):
            df_work[date_column] = pd.to_datetime(df_work[date_column], errors='coerce', format='mixed')
        
        df_work = df_work.dropna(subset=[date_column, target_column])
        
        if df_work.empty or len(df_work) < 30:
            return None
        
        df_work['_doy'] = df_work[date_column].dt.dayofyear
        
        # Агрегация по дню года
        grouped = df_work.groupby('_doy')[target_column].agg(
            ['mean', 'median', 'std', 'min', 'max', 'count',
             lambda x: x.quantile(0.10),
             lambda x: x.quantile(0.25),
             lambda x: x.quantile(0.75),
             lambda x: x.quantile(0.90)]
        )
        grouped.columns = ['mean', 'median', 'std', 'min', 'max', 'count', 'q10', 'q25', 'q75', 'q90']
        
        # Заполняем пропущенные дни года (если не все 366 дней представлены)
        full_index = pd.RangeIndex(1, 367)
        grouped = grouped.reindex(full_index)
        
        # Интерполируем пропущенные дни
        grouped = grouped.interpolate(method='linear', limit_direction='both')
        
        # Сглаживаем скользящим окном (±7 дней) для устранения шума
        smoothing_window = 15  # 7 дней до + текущий + 7 дней после
        for col in ['mean', 'median', 'std', 'q10', 'q25', 'q75', 'q90', 'min', 'max']:
            # Оборачиваем данные для корректного сглаживания на границах года
            extended = pd.concat([grouped[col].iloc[-7:], grouped[col], grouped[col].iloc[:7]])
            smoothed = extended.rolling(window=smoothing_window, center=True, min_periods=3).mean()
            grouped[col] = smoothed.iloc[7:-7].values
        
        # Заполняем NaN в std нулями
        grouped['std'] = grouped['std'].fillna(0)
        
        profile = grouped.to_dict('index')
        self.climatology[target_column] = profile
        
        return profile
    
    def _get_climatology_value(self, target_column, dayofyear, field='mean'):
        """Получить значение из климатологического профиля"""
        profile = self.climatology.get(target_column, {})
        doy = max(1, min(366, dayofyear))
        day_profile = profile.get(doy, {})
        if isinstance(day_profile, dict):
            return day_profile.get(field, 0)
        return 0
    
    def _handle_anomalies(self, df, target_column):
        """Обработка аномалий (выбросов) в целевой переменной для предотвращения ошибок R²"""
        df_clean = df.copy()
        if target_column and target_column in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[target_column]):
            # 1. Сначала убираем технические заглушки (часто -9999 или 9999)
            df_clean.loc[df_clean[target_column] < -9000, target_column] = np.nan
            df_clean.loc[df_clean[target_column] > 90000, target_column] = np.nan
            
            # 2. Вычисляем границы для типичных экстремумов
            series = df_clean[target_column].dropna()
            if len(series) > 10:
                q1 = series.quantile(0.05)
                q9 = series.quantile(0.95)
                iqr = q9 - q1
                
                # Если данные не константа
                if iqr > 1e-5:
                    lower_bound = q1 - 2.5 * iqr
                    upper_bound = q9 + 2.5 * iqr
                    
                    # Ограничиваем выбросы
                    df_clean.loc[df_clean[target_column] < lower_bound, target_column] = lower_bound
                    df_clean.loc[df_clean[target_column] > upper_bound, target_column] = upper_bound
        return df_clean

    def prepare_time_features(self, df, date_column, target_column=None):
        """Подготовить временные признаки с согласованным набором"""
        df_prepared = df.copy()
        
        # Сначала очищаем аномалии
        if target_column:
            df_prepared = self._handle_anomalies(df_prepared, target_column)
        
        # Преобразуем дату в pandas Timestamp
        if not pd.api.types.is_datetime64_any_dtype(df_prepared[date_column]):
            df_prepared[date_column] = pd.to_datetime(df_prepared[date_column], errors='coerce', format='mixed')
        
        # Удаляем строки с невалидными датами
        df_prepared = df_prepared.dropna(subset=[date_column])
        
        if df_prepared.empty:
            return df_prepared
        
        # Сортируем по дате
        df_prepared = df_prepared.sort_values(date_column).reset_index(drop=True)
        
        # Базовые временные признаки
        df_prepared['year'] = df_prepared[date_column].dt.year
        df_prepared['month'] = df_prepared[date_column].dt.month
        df_prepared['day'] = df_prepared[date_column].dt.day
        df_prepared['dayofweek'] = df_prepared[date_column].dt.dayofweek
        df_prepared['dayofyear'] = df_prepared[date_column].dt.dayofyear
        
        # Циклические тригонометрические признаки
        df_prepared['month_sin'] = np.sin(2 * np.pi * df_prepared['month'] / 12)
        df_prepared['month_cos'] = np.cos(2 * np.pi * df_prepared['month'] / 12)
        df_prepared['day_sin'] = np.sin(2 * np.pi * df_prepared['day'] / 31)
        df_prepared['day_cos'] = np.cos(2 * np.pi * df_prepared['day'] / 31)
        
        # Нормализованный год (для захвата многолетних трендов)
        year_min = df_prepared['year'].min()
        year_max = df_prepared['year'].max()
        if year_max > year_min:
            df_prepared['year_normalized'] = (df_prepared['year'] - year_min) / (year_max - year_min)
        else:
            df_prepared['year_normalized'] = 0.5
        
        # Только если целевая переменная существует
        if target_column and target_column in df_prepared.columns:
            # Заполняем небольшие пропуски (летние дни), оставляем зиму как NaN
            if df_prepared[target_column].isna().any():
                df_prepared[target_column] = df_prepared[target_column].interpolate(method='linear', limit=7)
            
            # Вычисляем резидуалы (отклонение от климатологической нормы)
            if target_column in self.climatology:
                df_prepared['_clim_mean'] = df_prepared['dayofyear'].apply(
                    lambda doy: self._get_climatology_value(target_column, doy, 'mean')
                )
                df_prepared['_residual'] = df_prepared[target_column] - df_prepared['_clim_mean']
            else:
                df_prepared['_residual'] = df_prepared[target_column]
            
            # Лаги на резидуалах (согласованный набор)
            if len(df_prepared) > max(self.LAG_DAYS):
                for lag in self.LAG_DAYS:
                    df_prepared[f'residual_lag_{lag}'] = df_prepared['_residual'].shift(lag)
            
            # Скользящие средние на резидуалах
            if len(df_prepared) > max(self.MA_WINDOWS):
                for window in self.MA_WINDOWS:
                    df_prepared[f'residual_ma_{window}'] = df_prepared['_residual'].rolling(
                        window=min(window, len(df_prepared)), min_periods=1
                    ).mean()
        
        return df_prepared
    
    def train_model(self, df, target_column, date_column, model_type='xgboost'):
        """Обучить модель предсказания с гибридным подходом:
        1) Построить климатологический профиль
        2) Обучить XGBoost на резидуалах (отклонениях от климатологии)
        """
        
        try:
            # Шаг 1: Строим климатологический профиль
            st.info("📊 Строим климатологический профиль из исторических данных...")
            profile = self._build_climatology_profile(df, target_column, date_column)
            if profile is None:
                st.warning("⚠️ Не удалось построить климатологический профиль, используем базовый режим")
            
            # Шаг 2: Подготовка данных (теперь с резидуалами)
            df_prepared = self.prepare_time_features(df, date_column, target_column)
            
            if df_prepared.empty or len(df_prepared) < 30:
                raise ValueError(f"Недостаточно данных для обучения. Найдено только {len(df_prepared)} строк.")
            
            if target_column not in df_prepared.columns:
                raise ValueError(f"Целевая переменная '{target_column}' не найдена в данных")
            
            # Сохраняем диапазон годов для нормализации при прогнозе
            self.year_range[target_column] = {
                'min': int(df_prepared['year'].min()),
                'max': int(df_prepared['year'].max())
            }
            
            # Шаг 3: Формируем согласованный набор признаков
            feature_cols = [f for f in self.BASE_FEATURES if f in df_prepared.columns]
            
            # Добавляем лаги и MA на резидуалах
            for lag in self.LAG_DAYS:
                col = f'residual_lag_{lag}'
                if col in df_prepared.columns:
                    feature_cols.append(col)
            for window in self.MA_WINDOWS:
                col = f'residual_ma_{window}'
                if col in df_prepared.columns:
                    feature_cols.append(col)
            
            # Также добавляем другие числовые колонки (например, температура, осадки)
            for col in df_prepared.columns:
                if (col not in feature_cols and
                    col not in [target_column, date_column, 'source_sheet', 'year', 'day',
                                '_clim_mean', '_residual'] and
                    not col.startswith('residual_') and
                    not col.startswith(f'{target_column}_') and
                    pd.api.types.is_numeric_dtype(df_prepared[col]) and
                    col not in ['year_normalized', 'month', 'dayofyear', 'dayofweek',
                                'month_sin', 'month_cos', 'day_sin', 'day_cos']):
                    # Добавляем только если колонка имеет достаточно данных
                    if df_prepared[col].notna().sum() > len(df_prepared) * 0.5:
                        feature_cols.append(col)
            
            # Удаляем дубликаты
            feature_cols = list(dict.fromkeys(feature_cols))
            
            # Целевая переменная — резидуал (отклонение от климатологии)
            if '_residual' in df_prepared.columns and profile is not None:
                y = df_prepared['_residual']
                st.info("🎯 Модель обучается на **отклонениях от исторической нормы** (резидуалах)")
            else:
                y = df_prepared[target_column]
                st.info("🎯 Модель обучается на **абсолютных значениях** (климатологический профиль недоступен)")
            
            X = df_prepared[feature_cols]
            
            # Удаляем строки где целевая переменная NaN
            valid_indices = y.notna() & X.notna().all(axis=1)
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) < 20:
                raise ValueError(f"Слишком мало данных после очистки: {len(X)} строк")
            
            # Сохраняем имена признаков
            self.feature_names[target_column] = feature_cols.copy()
            
            st.info(f"📋 Используется {len(feature_cols)} признаков, {len(X)} строк данных")
            
            # Разделение данных
            split_ratio = 0.7 if len(X) < 50 else 0.8
            split_idx = int(len(X) * split_ratio)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if len(X_train) < 10 or len(X_test) < 5:
                raise ValueError("Недостаточно данных для обучения/тестирования")
            
            # Заполнение пропущенных значений
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Масштабирование
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            scaler.feature_names_in_ = feature_cols
            imputer.feature_names_in_ = feature_cols
            
            # Настройка модели XGBoost
            if len(X_train) < 100:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42
                )
            
            # Обучение
            model.fit(X_train_scaled, y_train)
            
            # Оценка на резидуалах
            residual_score = model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0.0
            
            # Если обучались на резидуалах, пересчитаем R² для абсолютных значений
            if '_residual' in df_prepared.columns and profile is not None:
                y_test_pred_residual = model.predict(X_test_scaled)
                # Восстанавливаем абсолютные значения для тестовых данных
                test_doys = df_prepared.loc[valid_indices, 'dayofyear'].iloc[split_idx:]
                clim_means = test_doys.apply(
                    lambda doy: self._get_climatology_value(target_column, doy, 'mean')
                ).values
                y_test_abs_pred = y_test_pred_residual + clim_means
                y_test_abs_actual = df_prepared.loc[valid_indices, target_column].iloc[split_idx:].values
                
                # R² для абсолютных значений
                # Проверка на наличие аномальных значений (Inf/NaN) перед расчетом
                mask = np.isfinite(y_test_abs_actual) & np.isfinite(y_test_abs_pred)
                if mask.any():
                    y_act = y_test_abs_actual[mask]
                    y_pre = y_test_abs_pred[mask]
                    
                    ss_res = np.sum((y_act - y_pre) ** 2)
                    ss_tot = np.sum((y_act - np.mean(y_act)) ** 2)
                    score = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
                    
                    # Ограничиваем экстремально отрицательные значения для корректного отображения
                    if score < -1e6:
                        st.warning(f"⚠️ Внимание: Модель показывает крайне низкую точность (R²={score:.2e}). Возможно, в данных присутствуют аномалии или недостаточно исторических примеров.")
                        score = max(score, -999.999)
                else:
                    score = 0.0
                
                st.info(f"📈 R² на резидуалах: {residual_score:.3f} | R² для абсолютных значений: {score:.3f}")
            else:
                score = residual_score
            
            # Сохраняем модель и преобразователи
            self.models[target_column] = model
            self.scalers[target_column] = scaler
            self.imputers[target_column] = imputer
            
            # Важность признаков
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[target_column] = importance
            
            return score
            
        except Exception as e:
            raise ValueError(f"Ошибка обучения модели: {str(e)}")
    
    def predict_for_next_year(self, df, target_column, date_column):
        """Сделать прогноз на следующий год с использованием климатологии + XGBoost"""
        
        try:
            # Проверяем, обучена ли модель
            if target_column not in self.models:
                score = self.train_model(df, target_column, date_column)
                st.info(f"✅ Модель '{target_column}' обучена (R²={score:.3f})")
            
            # Подготавливаем данные
            df_prepared = self.prepare_time_features(df, date_column, target_column)
            
            if df_prepared.empty:
                raise ValueError("Нет данных для прогноза")
            
            # Последняя дата в данных
            last_date = df_prepared[date_column].max()
            next_year = last_date.year + 1
            
            # Создаем даты для следующего года
            start_date = pd.Timestamp(f"{next_year}-01-01")
            end_date = pd.Timestamp(f"{next_year}-12-31")
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Получаем историю резидуалов для лагов
            has_climatology = target_column in self.climatology
            if has_climatology and '_residual' in df_prepared.columns:
                residual_history = list(df_prepared['_residual'].dropna().values)
            else:
                residual_history = list(df_prepared[target_column].dropna().values)
            
            # Прогнозируем
            predictions = []
            predicted_residuals = []
            
            for i, forecast_date in enumerate(future_dates):
                # Создаем признаки для этой даты
                features = self._create_features_for_date(
                    forecast_date, 
                    target_column, 
                    residual_history, 
                    predicted_residuals
                )
                
                # Преобразуем в DataFrame с правильными именами признаков
                features_df = pd.DataFrame([features])
                
                expected_features = self.feature_names.get(target_column, [])
                if not expected_features:
                    expected_features = list(features.keys())
                
                for feat in expected_features:
                    if feat not in features_df.columns:
                        features_df[feat] = 0
                
                features_df = features_df[expected_features]
                
                # Обработка и предсказание
                features_imputed = self.imputers[target_column].transform(features_df)
                features_scaled = self.scalers[target_column].transform(features_imputed)
                predicted_residual = self.models[target_column].predict(features_scaled)[0]
                
                if np.isnan(float(predicted_residual)):
                    predicted_residual = 0.0
                
                # Восстанавливаем абсолютное значение
                doy = forecast_date.dayofyear
                if has_climatology:
                    clim_mean = self._get_climatology_value(target_column, doy, 'mean')
                    prediction = clim_mean + predicted_residual
                else:
                    clim_mean = 0
                    prediction = predicted_residual
                
                predictions.append(prediction)
                predicted_residuals.append(predicted_residual)
            
            # Формируем результат с климатологической информацией
            daily_predictions = []
            for i, (date, value) in enumerate(zip(future_dates, predictions)):
                doy = date.dayofyear
                pred_entry = {
                    'date': date,
                    'value': float(value),
                    'formatted_date': date.strftime('%d.%m.%Y'),
                    'day_of_year': doy,
                    'month': date.month,
                    'month_name': date.strftime('%B'),
                    'day_name': date.strftime('%A'),
                    'week_number': date.isocalendar()[1]
                }
                
                # Добавляем климатологические данные
                if has_climatology:
                    pred_entry['clim_mean'] = float(self._get_climatology_value(target_column, doy, 'mean'))
                    pred_entry['clim_q10'] = float(self._get_climatology_value(target_column, doy, 'q10'))
                    pred_entry['clim_q90'] = float(self._get_climatology_value(target_column, doy, 'q90'))
                    pred_entry['clim_min'] = float(self._get_climatology_value(target_column, doy, 'min'))
                    pred_entry['clim_max'] = float(self._get_climatology_value(target_column, doy, 'max'))
                    pred_entry['deviation'] = float(value - pred_entry['clim_mean'])
                
                daily_predictions.append(pred_entry)
            
            # Анализ по месяцам
            monthly_stats = {}
            for pred in daily_predictions:
                month_key = f"{pred['month']:02d}"
                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {
                        'month_name': pred['month_name'],
                        'values': [],
                        'dates': []
                    }
                monthly_stats[month_key]['values'].append(pred['value'])
                monthly_stats[month_key]['dates'].append(pred['date'])
            
            # Рассчитываем статистику по месяцам
            monthly_summary = {}
            for month, data in monthly_stats.items():
                if data['values']:
                    monthly_summary[month] = {
                        'month_name': data['month_name'],
                        'average': float(np.mean(data['values'])),
                        'min': float(np.min(data['values'])),
                        'max': float(np.max(data['values'])),
                        'std': float(np.std(data['values']))
                    }
            
            # Находим ключевые даты
            if daily_predictions:
                values = [p['value'] for p in daily_predictions]
                max_idx = np.argmax(values)
                min_idx = np.argmin(values)
                
                key_dates = {
                    'max_value': daily_predictions[max_idx],
                    'min_value': daily_predictions[min_idx],
                    'first_day': daily_predictions[0],
                    'last_day': daily_predictions[-1]
                }
            else:
                key_dates = {}
            
            # Историческая статистика
            historical_values = df_prepared[target_column].dropna().values
            historical_stats = {
                'last_date': last_date.strftime('%d.%m.%Y'),
                'last_value': float(historical_values[-1]) if len(historical_values) > 0 else None,
                'historical_mean': float(np.mean(historical_values)) if len(historical_values) > 0 else None,
                'historical_std': float(np.std(historical_values)) if len(historical_values) > 1 else None,
                'data_years': f"{df_prepared[date_column].dt.year.min()}-{df_prepared[date_column].dt.year.max()}"
            }
            
            # Анализ сезонов
            seasons_analysis = self._analyze_seasons(daily_predictions)
            
            # Формируем итоговый вывод
            conclusion = self._generate_conclusion(target_column, key_dates, monthly_summary, seasons_analysis)
            
            return {
                'year': next_year,
                'daily_predictions': daily_predictions,
                'monthly_summary': monthly_summary,
                'key_dates': key_dates,
                'historical_stats': historical_stats,
                'seasons_analysis': seasons_analysis,
                'conclusion': conclusion,
                'total_predictions': len(daily_predictions)
            }
            
        except Exception as e:
            raise ValueError(f"Ошибка прогноза на следующий год: {str(e)}")
    
    def _create_features_for_date(self, date, target_column, residual_history, predicted_residuals):
        """Создать признаки для конкретной даты (согласованно с train_model).
        
        residual_history: список исторических резидуалов (или значений если нет климатологии)
        predicted_residuals: список уже предсказанных резидуалов
        """
        
        # Нормализованный год
        yr = self.year_range.get(target_column, {'min': date.year, 'max': date.year})
        year_min, year_max = yr['min'], yr['max']
        if year_max > year_min:
            year_normalized = (date.year - year_min) / (year_max - year_min)
        else:
            year_normalized = 0.5
        
        # Базовые временные признаки (точно как в prepare_time_features)
        features = {
            'month': date.month,
            'dayofyear': date.dayofyear,
            'dayofweek': date.dayofweek,
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'day_sin': np.sin(2 * np.pi * date.day / 31),
            'day_cos': np.cos(2 * np.pi * date.day / 31),
            'year_normalized': year_normalized,
        }
        
        # Объединяем историю резидуалов и предсказанные
        all_residuals = list(residual_history) + predicted_residuals
        total_len = len(all_residuals)
        
        # Лаги на резидуалах (согласованный набор)
        for lag in self.LAG_DAYS:
            lag_key = f'residual_lag_{lag}'
            if total_len >= lag:
                features[lag_key] = all_residuals[-lag]
            else:
                features[lag_key] = all_residuals[-1] if all_residuals else 0
        
        # Скользящие средние на резидуалах
        for window in self.MA_WINDOWS:
            ma_key = f'residual_ma_{window}'
            if total_len >= window:
                features[ma_key] = float(np.mean(all_residuals[-window:]))
            else:
                features[ma_key] = float(np.mean(all_residuals)) if all_residuals else 0
        
        # Добавляем фиктивные признаки, которые могли быть при обучении
        for col in self.feature_names.get(target_column, []):
            if col not in features:
                features[col] = 0
        
        return features
    
    def _analyze_seasons(self, predictions):
        """Анализ по сезонам"""
        seasons = {
            'зима': [12, 1, 2],
            'весна': [3, 4, 5],
            'лето': [6, 7, 8],
            'осень': [9, 10, 11]
        }
        
        analysis = {}
        for season_name, months in seasons.items():
            season_values = [p['value'] for p in predictions if p['month'] in months]
            if season_values:
                analysis[season_name] = {
                    'average': float(np.mean(season_values)),
                    'min': float(np.min(season_values)),
                    'max': float(np.max(season_values)),
                    'trend': 'выше нормы' if np.mean(season_values) > 0 else 'ниже нормы'
                }
        
        return analysis
    
    def _generate_conclusion(self, target_column, key_dates, monthly_summary, seasons_analysis):
        """Сгенерировать итоговый вывод"""
        
        conclusions = []
        
        if not key_dates:
            return ["Недостаточно данных для формирования вывода"]
        
        # Основная информация
        max_info = key_dates.get('max_value', {})
        min_info = key_dates.get('min_value', {})
        
        if max_info and min_info:
            conclusions.append("**📅 Основные экстремумы:**")
            conclusions.append(f"- Максимум: {max_info.get('value', 0):.1f} ({max_info.get('formatted_date', 'N/A')})")
            conclusions.append(f"- Минимум: {min_info.get('value', 0):.1f} ({min_info.get('formatted_date', 'N/A')})")
        
        # Анализ по сезонам
        if seasons_analysis:
            conclusions.append("**🌦️ Сезонный анализ:**")
            for season, stats in seasons_analysis.items():
                conclusions.append(f"- {season.capitalize()}: среднее {stats['average']:.1f} (от {stats['min']:.1f} до {stats['max']:.1f})")
        
        # Специфичный анализ для разных типов данных
        target_lower = target_column.lower()
        
        if 'температур' in target_lower:
            # Анализ для температуры
            if 'max_value' in key_dates and key_dates['max_value'].get('value', 0) > 25:
                conclusions.append("**🔥 Жаркие периоды:** Ожидается жаркое лето с температурами выше 25°C")
            
            if 'min_value' in key_dates and key_dates['min_value'].get('value', 0) < -10:
                conclusions.append("**❄️ Холодные периоды:** Ожидаются морозы ниже -10°C")
            
            # Рекомендации
            conclusions.append("**💡 Рекомендации:**")
            conclusions.append("- Подготовьте системы кондиционирования к летнему периоду")
            conclusions.append("- Утеплите помещения к зимним морозам")
        
        elif 'осадк' in target_lower:
            # Анализ для осадков
            total_precip = sum([p.get('value', 0) for p in key_dates.values() if isinstance(p, dict)])
            
            if total_precip > 1000:
                conclusions.append("**🌧️ Обильные осадки:** Ожидается высокий уровень осадков")
                conclusions.append("**💡 Рекомендации:**")
                conclusions.append("- Проверьте дренажные системы")
                conclusions.append("- Подготовьтесь к возможным паводкам")
            else:
                conclusions.append("**☀️ Умеренные осадки:** Уровень осадков в пределах нормы")
        
        elif 'уровен' in target_lower or 'вод' in target_lower:
            # Анализ для уровня воды
            conclusions.append("**🌊 Гидрологический прогноз:**")
            conclusions.append("- Следите за изменениями уровня воды в паводковый период")
            conclusions.append("**💡 Рекомендации:**")
            conclusions.append("- Проверьте защитные сооружения")
            conclusions.append("- Подготовьте план действий при подъеме уровня воды")
        
        else:
            # Общие рекомендации
            conclusions.append("**📊 Общий анализ:** Прогноз основан на исторических данных")
            conclusions.append("**💡 Рекомендации:**")
            conclusions.append("- Регулярно обновляйте данные для повышения точности")
            conclusions.append("- Сравнивайте прогноз с фактическими наблюдениями")
        
        return conclusions
    
    def predict_for_specific_date(self, df, target_column, date_column, specific_date_str):
        """Предсказать значение на конкретную дату"""
        
        try:
            # Преобразуем дату
            specific_date = pd.to_datetime(specific_date_str, errors='coerce', format='mixed')
            if pd.isna(specific_date):
                raise ValueError(f"Неверный формат даты: {specific_date_str}")
            
            # Обучаем модель если еще не обучена
            if target_column not in self.models:
                score = self.train_model(df, target_column, date_column)
                st.info(f"Модель '{target_column}' обучена (R²={score:.3f})")
            
            # Получаем прогноз на следующий год
            forecast_result = self.predict_for_next_year(df, target_column, date_column)
            
            # Ищем ближайшую дату в прогнозе
            predictions = forecast_result['daily_predictions']
            
            if not predictions:
                raise ValueError("Нет данных прогноза")
            
            # Ищем точное совпадение
            for pred in predictions:
                if pred['date'].date() == specific_date.date():
                    return {
                        'requested_date': specific_date.strftime('%d.%m.%Y'),
                        'exact_match': True,
                        'value': pred['value'],
                        'date': pred['formatted_date'],
                        'confidence': 'очень высокая',
                        'additional_info': self._get_date_analysis(pred, forecast_result)
                    }
            
            # Если точного совпадения нет, находим ближайшую дату
            closest_pred = None
            min_diff = float('inf')
            
            for pred in predictions:
                diff = abs((pred['date'] - specific_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_pred = pred
            
            if closest_pred:
                return {
                    'requested_date': specific_date.strftime('%d.%m.%Y'),
                    'exact_match': False,
                    'value': closest_pred['value'],
                    'date': closest_pred['formatted_date'],
                    'days_difference': min_diff,
                    'confidence': 'высокая' if min_diff <= 3 else 'средняя',
                    'additional_info': self._get_date_analysis(closest_pred, forecast_result)
                }
            else:
                raise ValueError(f"Не удалось найти прогноз для даты {specific_date_str}")
                
        except Exception as e:
            raise ValueError(f"Ошибка прогноза на конкретную дату: {str(e)}")
    
    def _get_date_analysis(self, prediction, forecast_result):
        """Получить анализ для конкретной даты"""
        analysis = []
        
        # Сравнение с историческим средним
        hist_mean = forecast_result.get('historical_stats', {}).get('historical_mean')
        if hist_mean:
            diff = prediction['value'] - hist_mean
            if diff > 0:
                analysis.append(f"Выше исторического среднего на {diff:.1f}")
            else:
                analysis.append(f"Ниже исторического среднего на {abs(diff):.1f}")
        
        # Сравнение с сезонными нормами
        month = prediction['month']
        monthly_summary = forecast_result.get('monthly_summary', {})
        month_key = f"{month:02d}"
        
        if month_key in monthly_summary:
            month_avg = monthly_summary[month_key]['average']
            diff_month = prediction['value'] - month_avg
            if diff_month > 0:
                analysis.append(f"Выше среднего для {prediction['month_name']} на {diff_month:.1f}")
            else:
                analysis.append(f"Ниже среднего для {prediction['month_name']} на {abs(diff_month):.1f}")
        
        # Погодные условия
        if 'температур' in prediction.get('target_column', '').lower():
            temp = prediction['value']
            if temp > 25:
                analysis.append("Ожидается жаркая погода")
            elif temp < 0:
                analysis.append("Ожидается морозная погода")
            elif temp < 10:
                analysis.append("Ожидается прохладная погода")
            else:
                analysis.append("Ожидается комфортная температура")
        
        elif 'осадк' in prediction.get('target_column', '').lower():
            precip = prediction['value']
            if precip > 10:
                analysis.append("Ожидаются сильные осадки")
            elif precip > 5:
                analysis.append("Ожидаются умеренные осадки")
            elif precip > 0:
                analysis.append("Возможны слабые осадки")
            else:
                analysis.append("Осадки не ожидаются")
        
        return analysis
    
    def create_detailed_forecast_report(self, df, target_column, date_column):
        """Создать детальный отчет с прогнозами"""
        
        try:
            # Получаем прогноз на следующий год
            forecast_result = self.predict_for_next_year(df, target_column, date_column)
            
            if not forecast_result['daily_predictions']:
                raise ValueError("Не удалось получить прогнозы")
            
            # Формируем отчет
            report = {
                'target': target_column,
                'forecast_year': forecast_result['year'],
                'summary': '',
                'key_dates': [],
                'monthly_analysis': [],
                'seasons_analysis': [],
                'conclusions': forecast_result.get('conclusion', []),
                'historical_context': forecast_result.get('historical_stats', {})
            }
            
            # Основная сводка
            key_dates = forecast_result.get('key_dates', {})
            
            if key_dates and 'max_value' in key_dates and 'min_value' in key_dates:
                max_info = key_dates['max_value']
                min_info = key_dates['min_value']
                
                report['summary'] = (
                    f"**Прогноз {target_column} на {forecast_result['year']} год**\n\n"
                    f"📈 **Максимальное значение:** {max_info.get('value', 'N/A'):.1f} "
                    f"({max_info.get('formatted_date', 'N/A')})\n"
                    f"📉 **Минимальное значение:** {min_info.get('value', 'N/A'):.1f} "
                    f"({min_info.get('formatted_date', 'N/A')})\n"
                    f"📅 **Период прогноза:** 01.01.{forecast_result['year']} - 31.12.{forecast_result['year']}\n"
                    f"📊 **Исторические данные:** {forecast_result.get('historical_stats', {}).get('data_years', 'N/A')}"
                )
            
            # Ключевые даты
            important_dates = [
                ("01.01", "Новый год"),
                ("01.03", "Первый день весны"),
                ("01.06", "Первый день лета"),
                ("01.09", "Первый день осени"),
                ("01.12", "Первый день зимы"),
                ("01.07", "Середина года"),
                ("31.12", "Канун Нового года")
            ]
            
            for date_str, description in important_dates:
                target_date = f"{date_str}.{forecast_result['year']}"
                try:
                    result = self.predict_for_specific_date(df, target_column, date_column, target_date)
                    if result:
                        report['key_dates'].append({
                            'date': result['date'],
                            'description': description,
                            'value': f"{result['value']:.1f}",
                            'confidence': result.get('confidence', 'N/A')
                        })
                except Exception as e:
                    continue
            
            # Анализ по месяцам
            monthly_summary = forecast_result.get('monthly_summary', {})
            for month, stats in monthly_summary.items():
                report['monthly_analysis'].append({
                    'month': stats['month_name'],
                    'average': f"{stats['average']:.1f}",
                    'range': f"{stats['min']:.1f} - {stats['max']:.1f}",
                    'stability': 'высокая' if stats['std'] < stats['average'] * 0.1 else 'средняя'
                })
            
            # Анализ по сезонам
            seasons_analysis = forecast_result.get('seasons_analysis', {})
            for season, stats in seasons_analysis.items():
                report['seasons_analysis'].append({
                    'season': season.capitalize(),
                    'average': f"{stats['average']:.1f}",
                    'trend': stats['trend']
                })
            
            return report
            
        except Exception as e:
            raise ValueError(f"Ошибка создания отчета: {str(e)}")
    
    def create_visual_forecast(self, df, target_column, date_column):
        """Создать визуализацию прогноза с климатологическим профилем"""
        
        try:
            forecast_result = self.predict_for_next_year(df, target_column, date_column)
            
            if not forecast_result['daily_predictions']:
                return None
            
            # Подготавливаем данные для графика
            predictions = forecast_result['daily_predictions']
            forecast_dates = [pred['date'] for pred in predictions]
            forecast_values = [pred['value'] for pred in predictions]
            
            # Создаем график
            fig = go.Figure()
            
            # Добавляем климатологический диапазон (q10-q90) если доступен
            has_climatology = 'clim_mean' in predictions[0]
            if has_climatology:
                clim_q10 = [pred.get('clim_q10', pred['value']) for pred in predictions]
                clim_q90 = [pred.get('clim_q90', pred['value']) for pred in predictions]
                clim_mean = [pred.get('clim_mean', pred['value']) for pred in predictions]
                
                # Полоса исторического диапазона (q10-q90)
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=clim_q90 + clim_q10[::-1],
                    fill='toself',
                    fillcolor='rgba(100, 149, 237, 0.15)',
                    line=dict(color='rgba(100, 149, 237, 0)'),
                    name='Ист. диапазон (10-90%)',
                    hoverinfo='skip',
                    showlegend=True
                ))
                
                # Линия климатологического среднего
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=clim_mean,
                    mode='lines',
                    name='Ист. среднее',
                    line=dict(color='rgba(100, 149, 237, 0.6)', width=2, dash='dot'),
                ))
            
            # Прогноз — главная линия
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name='Прогноз',
                line=dict(color='#E53935', width=3, shape='spline'),
            ))
            
            # Добавляем маркеры для ключевых дат
            key_dates = forecast_result.get('key_dates', {})
            if 'max_value' in key_dates:
                max_info = key_dates['max_value']
                fig.add_trace(go.Scatter(
                    x=[max_info['date']],
                    y=[max_info['value']],
                    mode='markers+text',
                    name='Максимум',
                    marker=dict(size=14, color='red', symbol='triangle-up'),
                    text=[f"Макс: {max_info['value']:.1f}"],
                    textposition="top center"
                ))
            
            if 'min_value' in key_dates:
                min_info = key_dates['min_value']
                fig.add_trace(go.Scatter(
                    x=[min_info['date']],
                    y=[min_info['value']],
                    mode='markers+text',
                    name='Минимум',
                    marker=dict(size=14, color='blue', symbol='triangle-down'),
                    text=[f"Мин: {min_info['value']:.1f}"],
                    textposition="bottom center"
                ))
            
            # Добавляем пороги, если они установлены
            if hasattr(self, 'thresholds_set') and self.thresholds_set:
                high_val = getattr(self, 'high_threshold', None)
                danger_val = getattr(self, 'danger_threshold', None)
                
                if high_val is not None:
                    fig.add_hline(y=high_val, line_dash="dash", line_color="orange",
                                annotation_text=f"Высокий уровень ({high_val})", 
                                annotation_position="bottom right")
                
                if danger_val is not None:
                    fig.add_hline(y=danger_val, line_dash="dash", line_color="red",
                                annotation_text=f"Опасный уровень ({danger_val})", 
                                annotation_position="top right")
            
            # Настройка графика
            clim_note = " (с историческим профилем)" if has_climatology else ""
            fig.update_layout(
                title=f'Прогноз {target_column} на {forecast_result["year"]} год{clim_note}',
                xaxis_title='Дата',
                yaxis_title=target_column,
                hovermode='x unified',
                template='plotly_white',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Ошибка создания визуализации: {str(e)}")
            return None
    
    def analyze_seasonal_patterns(self, df, target_column, date_column):
        """Анализ сезонных паттернов"""
        
        try:
            df_prepared = self.prepare_time_features(df, date_column, target_column)
            df_prepared = df_prepared.dropna(subset=[target_column])
            
            if df_prepared.empty or len(df_prepared) < 30:
                return None
            
            # Создаем график
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Распределение по месяцам',
                    'Средние по годам',
                    'Тренд за все время',
                    'Сезонные паттерны'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. Распределение по месяцам
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                          'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
            
            for month in range(1, 13):
                month_values = df_prepared[df_prepared['month'] == month][target_column]
                if not month_values.empty:
                    fig.add_trace(
                        go.Box(
                            y=month_values,
                            name=month_names[month-1]
                        ),
                        row=1, col=1
                    )
            
            # 2. Средние по годам
            yearly_avg = df_prepared.groupby('year')[target_column].mean()
            if not yearly_avg.empty:
                fig.add_trace(
                    go.Bar(
                        x=yearly_avg.index,
                        y=yearly_avg.values,
                        name='Среднее по годам'
                    ),
                    row=1, col=2
                )
            
            # 3. Тренд
            if len(df_prepared) > 10:
                df_sorted = df_prepared.sort_values(date_column)
                window_size = min(30, len(df_sorted) // 10)
                df_sorted['smoothed'] = df_sorted[target_column].rolling(window=window_size, center=True).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[date_column],
                        y=df_sorted['smoothed'],
                        mode='lines',
                        name='Тренд',
                        line=dict(color='red', width=2, shape='spline')
                    ),
                    row=2, col=1
                )
            
            # 4. Сезонные паттерны
            if 'dayofyear' in df_prepared.columns:
                doy_avg = df_prepared.groupby('dayofyear')[target_column].mean()
                if not doy_avg.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=doy_avg.index,
                            y=doy_avg.values,
                            mode='lines',
                            name='Сезонный паттерн',
                            line=dict(color='green', width=2, shape='spline')
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=700,
                title_text=f"Сезонный анализ: {target_column}",
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Ошибка анализа сезонных паттернов: {str(e)}")
            return None
    
    def get_forecast_metrics(self, df, target_column, date_column):
        """Получить метрики качества прогноза"""
        try:
            # Простые метрики
            df_prepared = self.prepare_time_features(df, date_column, target_column)
            df_prepared = df_prepared.dropna(subset=[target_column])
            
            if len(df_prepared) < 50:
                return {}
            
            # Рассчитываем базовые статистики
            values = df_prepared[target_column].values
            metrics = {
                'historical_mean': float(np.mean(values)),
                'historical_std': float(np.std(values)),
                'historical_min': float(np.min(values)),
                'historical_max': float(np.max(values)),
                'data_points': len(values),
                'data_years': f"{df_prepared[date_column].dt.year.min()}-{df_prepared[date_column].dt.year.max()}"
            }
            
            return metrics
            
        except Exception as e:
            return {}
    
    def save_model(self, target_column, filename):
        """Сохранить обученную модель"""
        if target_column in self.models:
            model_data = {
                'model': self.models[target_column],
                'scaler': self.scalers.get(target_column),
                'imputer': self.imputers.get(target_column),
                'feature_names': self.feature_names.get(target_column),
                'feature_importance': self.feature_importance.get(target_column)
            }
            joblib.dump(model_data, filename)
            return True
        return False

class RiverDataPredictor(TimeSeriesPredictor):
    """Специализированный предсказатель для речных данных с учетом сезонности"""
    
    def __init__(self):
        super().__init__()
        # Добавляем атрибуты для порогового анализа
        self.high_threshold = None
        self.danger_threshold = None
        self.thresholds_set = False
        # Период открытой воды (май-октябрь)
        self.open_water_months = [5, 6, 7, 8, 9, 10]  # Май-Октябрь
        self.winter_months = [11, 12, 1, 2, 3, 4]      # Ноябрь-Апрель
        
    def set_thresholds(self, high, danger):
        """Установить пороговые значения"""
        self.high_threshold = high
        self.danger_threshold = danger
        self.thresholds_set = True
        return True
    
    def analyze_with_thresholds(self, df, target_column, date_column):
        """Анализ с учетом порогов - ТОЛЬКО для сезона открытой воды"""
        if not self.thresholds_set:
            raise ValueError("Пороги не установлены. Используйте set_thresholds()")
        
        # 1. Получаем прогноз только на сезон открытой воды
        forecast = self.predict_for_water_season(df, target_column, date_column)
        
        # 2. Анализируем относительно порогов
        threshold_analysis = self._analyze_thresholds(forecast)
        
        # 3. Классифицируем дни по опасности
        danger_classification = self._classify_danger_days(forecast)
        
        # 4. Формируем отчет
        report = {
            **forecast,
            'threshold_analysis': threshold_analysis,
            'danger_classification': danger_classification,
            'high_threshold': self.high_threshold,
            'danger_threshold': self.danger_threshold
        }
        
        return report
    
    def predict_for_water_season(self, df, target_column, date_column, year=None):
        """Прогноз только на сезон открытой воды (май-октябрь) с климатологией"""
        
        try:
            # Подготавливаем данные
            df_prepared = self.prepare_time_features(df, date_column, target_column)
            
            if df_prepared.empty:
                raise ValueError("Нет данных для прогноза")
            
            # Определяем год прогноза
            last_date = df_prepared[date_column].max()
            if year is None:
                forecast_year = last_date.year + 1
            else:
                forecast_year = year
            
            # Создаем даты только для периода открытой воды
            start_date = pd.Timestamp(f"{forecast_year}-05-01")
            end_date = pd.Timestamp(f"{forecast_year}-10-31")
            
            current_date = pd.Timestamp.now()
            if forecast_year == current_date.year and current_date > end_date:
                forecast_year += 1
                start_date = pd.Timestamp(f"{forecast_year}-05-01")
                end_date = pd.Timestamp(f"{forecast_year}-10-31")
            
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Обучаем модель если еще не обучена (на всех данных, не только сезонных)
            if target_column not in self.models:
                st.info(f"🔄 Обучаю модель с климатологическим профилем...")
                score = self.train_model(df, target_column, date_column)
                if score is None:
                    score = 0.0
                st.success(f"✅ Модель обучена (R²={score:.3f})")
            
            # Получаем историю резидуалов
            has_climatology = target_column in self.climatology
            if has_climatology and '_residual' in df_prepared.columns:
                residual_history = list(df_prepared['_residual'].dropna().values)
            else:
                residual_history = list(df_prepared[target_column].dropna().values)
            
            # Прогнозируем
            predictions = []
            predicted_residuals = []
            
            for i, forecast_date in enumerate(future_dates):
                if forecast_date.month not in self.open_water_months:
                    continue
                
                # Используем согласованный метод создания признаков
                features = self._create_features_for_date(
                    forecast_date, 
                    target_column, 
                    residual_history, 
                    predicted_residuals
                )
                
                features_df = pd.DataFrame([features])
                expected_features = self.feature_names.get(target_column, [])
                if not expected_features:
                    expected_features = list(features.keys())
                
                for feat in expected_features:
                    if feat not in features_df.columns:
                        features_df[feat] = 0
                
                features_df = features_df[expected_features]
                
                try:
                    features_imputed = self.imputers[target_column].transform(features_df)
                    features_scaled = self.scalers[target_column].transform(features_imputed)
                    predicted_residual = self.models[target_column].predict(features_scaled)[0]
                    
                    if np.isnan(float(predicted_residual)):
                        raise ValueError("Прогноз вернул NaN")
                    
                    # Восстанавливаем абсолютное значение
                    doy = forecast_date.dayofyear
                    if has_climatology:
                        clim_mean = self._get_climatology_value(target_column, doy, 'mean')
                        prediction = clim_mean + predicted_residual
                    else:
                        prediction = predicted_residual
                    
                    predictions.append(prediction)
                    predicted_residuals.append(predicted_residual)
                    
                except Exception as e:
                    # Fallback — используем климатологию
                    doy = forecast_date.dayofyear
                    if has_climatology:
                        prediction = self._get_climatology_value(target_column, doy, 'mean')
                    elif residual_history:
                        prediction = float(np.mean(residual_history[-30:]))
                    else:
                        prediction = 0
                    predictions.append(prediction)
                    predicted_residuals.append(0)
            
            # Формируем результат с климатологическими данными
            daily_predictions = []
            for i, (date, value) in enumerate(zip(future_dates, predictions)):
                doy = date.dayofyear
                pred_entry = {
                    'date': date,
                    'value': float(value),
                    'formatted_date': date.strftime('%d.%m.%Y'),
                    'day_of_year': doy,
                    'month': date.month,
                    'month_name': date.strftime('%B'),
                    'day_name': date.strftime('%A'),
                    'week_number': date.isocalendar()[1],
                    'season': 'open_water'
                }
                
                if has_climatology:
                    pred_entry['clim_mean'] = float(self._get_climatology_value(target_column, doy, 'mean'))
                    pred_entry['clim_q10'] = float(self._get_climatology_value(target_column, doy, 'q10'))
                    pred_entry['clim_q90'] = float(self._get_climatology_value(target_column, doy, 'q90'))
                    pred_entry['clim_min'] = float(self._get_climatology_value(target_column, doy, 'min'))
                    pred_entry['clim_max'] = float(self._get_climatology_value(target_column, doy, 'max'))
                    pred_entry['deviation'] = float(value - pred_entry['clim_mean'])
                
                daily_predictions.append(pred_entry)
            
            # Анализ по месяцам
            monthly_stats = {}
            for pred in daily_predictions:
                if pred['month'] in self.open_water_months:
                    month_key = f"{pred['month']:02d}"
                    if month_key not in monthly_stats:
                        monthly_stats[month_key] = {
                            'month_name': pred['month_name'],
                            'values': [],
                            'dates': []
                        }
                    monthly_stats[month_key]['values'].append(pred['value'])
                    monthly_stats[month_key]['dates'].append(pred['date'])
            
            monthly_summary = {}
            for month, data in monthly_stats.items():
                if data['values']:
                    monthly_summary[month] = {
                        'month_name': data['month_name'],
                        'average': float(np.mean(data['values'])),
                        'min': float(np.min(data['values'])),
                        'max': float(np.max(data['values'])),
                        'std': float(np.std(data['values'])),
                        'data_points': len(data['values'])
                    }
            
            # Ключевые даты
            if daily_predictions:
                values = [p['value'] for p in daily_predictions]
                max_idx = np.argmax(values)
                min_idx = np.argmin(values)
                
                key_dates = {
                    'max_value': daily_predictions[max_idx],
                    'min_value': daily_predictions[min_idx],
                    'first_day': daily_predictions[0],
                    'last_day': daily_predictions[-1],
                    'season_start': daily_predictions[0]['date'],
                    'season_end': daily_predictions[-1]['date']
                }
            else:
                key_dates = {}
            
            # Фильтруем исторические данные для открытой воды
            historical_data = df_prepared[
                df_prepared[date_column].dt.month.isin(self.open_water_months)
            ].copy()
            
            hist_open_water = historical_data[target_column] if not historical_data.empty else pd.Series(dtype=float)
            historical_stats = {
                'last_date': last_date.strftime('%d.%m.%Y'),
                'last_value': float(hist_open_water.iloc[-1]) if len(hist_open_water) > 0 else None,
                'historical_mean': float(hist_open_water.mean()) if len(hist_open_water) > 0 else None,
                'historical_std': float(hist_open_water.std()) if len(hist_open_water) > 1 else None,
                'historical_min': float(hist_open_water.min()) if len(hist_open_water) > 0 else None,
                'historical_max': float(hist_open_water.max()) if len(hist_open_water) > 0 else None,
                'data_years': f"{df_prepared[date_column].dt.year.min()}-{df_prepared[date_column].dt.year.max()}",
                'season_months': self.open_water_months,
                'total_data_points': len(hist_open_water),
                'data_quality': 'хорошая' if len(hist_open_water) >= 100 else 'удовлетворительная' if len(hist_open_water) >= 30 else 'низкая'
            }
            
            # Анализ по периодам
            season_analysis = self._analyze_water_season(daily_predictions)
            
            # Вывод
            conclusion = self._generate_water_season_conclusion(
                target_column, key_dates, monthly_summary, season_analysis, forecast_year
            )
            
            return {
                'year': forecast_year,
                'season': 'open_water',
                'season_months': self.open_water_months,
                'season_period': f"{start_date.strftime('%d.%m')} - {end_date.strftime('%d.%m')}",
                'daily_predictions': daily_predictions,
                'monthly_summary': monthly_summary,
                'key_dates': key_dates,
                'historical_stats': historical_stats,
                'season_analysis': season_analysis,
                'conclusion': conclusion,
                'total_predictions': len(daily_predictions),
                'note': 'Прогноз только для сезона открытой воды (май-октябрь). Зимние месяцы (ноябрь-апрель) не анализируются.'
            }
            
        except Exception as e:
            st.error(f"❌ Ошибка прогноза на сезон открытой воды: {str(e)}")
            raise
    
    def _add_seasonal_features(self, df, date_column):
        """Добавить сезонные признаки для лучшего обучения модели"""
        df = df.copy()
        
        # Гидрологические сезоны внутри периода открытой воды
        df['hydro_season'] = df[date_column].apply(self._get_hydro_season)
        
        # Взаимодействие месяца и дня
        df['month_day_interaction'] = df[date_column].dt.month * df[date_column].dt.day
        
        # Признаки для циклических паттернов
        df['month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df[date_column].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df[date_column].dt.day / 31)
        
        return df
    
    def _get_hydro_season(self, date):
        """Определить гидрологический сезон для даты"""
        month = date.month
        
        if month == 5:  # Май - весенний паводок
            return 1
        elif month in [6, 7]:  # Июнь-Июль - летняя межень
            return 2
        elif month in [8, 9]:  # Август-Сентябрь - дождевые паводки
            return 3
        elif month == 10:  # Октябрь - осенний период
            return 4
        else:
            return 0  # Вне сезона (не должно быть)
    
    def _create_features_for_water_season_date(self, date, target_column, historical_values, predicted_values):
        """Создать признаки для конкретной даты с учетом сезонности"""
        
        # Базовые временные признаки
        features = {
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'dayofweek': date.dayofweek,
            'dayofyear': date.dayofyear,
            'week': date.isocalendar()[1],
            'quarter': (date.month - 1) // 3 + 1,
            'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'day_sin': np.sin(2 * np.pi * date.day / 31),
            'day_cos': np.cos(2 * np.pi * date.day / 31),
            'hydro_season': self._get_hydro_season(date),
            'month_day_interaction': date.month * date.day
        }
        
        # Объединяем исторические и предсказанные значения
        all_values = list(historical_values) + predicted_values
        total_len = len(all_values)
        
        # Добавляем лаговые признаки с учетом сезонности
        for lag in [1, 2, 3, 7, 14, 30]:  # Более короткие лаги для сезонных данных
            lag_key = f'{target_column}_lag_{lag}'
            if total_len >= lag:
                features[lag_key] = all_values[-lag]
            else:
                # Используем последнее доступное значение
                features[lag_key] = all_values[-1] if all_values else 0
        
        # Скользящие средние с окнами, подходящими для сезона
        for window in [3, 7, 14, 30]:
            ma_key = f'{target_column}_ma_{window}'
            if total_len >= window:
                features[ma_key] = float(np.mean(all_values[-window:]))
            else:
                features[ma_key] = float(np.mean(all_values)) if all_values else 0
        
        # Сезонные скользящие средние (среднее за тот же месяц в прошлом)
        if total_len >= 30:
            pass
            # Попробуем найти значения за тот же месяц в исторических данных
            month_values = [v for i, v in enumerate(all_values) if i < len(historical_values)]
            if len(month_values) >= 15:
                features[f'{target_column}_month_avg'] = float(np.mean(month_values[-15:]))
        
        # Добавляем фиктивные признаки, которые могли быть при обучении
        if target_column in self.feature_names:
            for col in self.feature_names[target_column]:
                if col not in features:
                    features[col] = 0
        
        return features
    
    def _analyze_water_season(self, predictions):
        """Анализ по периодам внутри сезона открытой воды"""
        periods = {
            'весенний_паводок': {'months': [5], 'name': 'Весенний паводок'},
            'летняя_межень': {'months': [6, 7], 'name': 'Летняя межень'},
            'летние_дождевые_паводки': {'months': [8, 9], 'name': 'Дождевые паводки'},
            'осенний_период': {'months': [10], 'name': 'Осенний период'}
        }
        
        analysis = {}
        for period_key, period_info in periods.items():
            period_values = [p['value'] for p in predictions if p['month'] in period_info['months']]
            if period_values:
                analysis[period_key] = {
                    'name': period_info['name'],
                    'average': float(np.mean(period_values)),
                    'min': float(np.min(period_values)),
                    'max': float(np.max(period_values)),
                    'range': float(np.max(period_values) - np.min(period_values)),
                    'std': float(np.std(period_values)),
                    'data_points': len(period_values),
                    'months': period_info['months'],
                    'trend': self._calculate_trend(period_values)
                }
        
        return analysis
    
    def _calculate_trend(self, values):
        """Рассчитать тренд для набора значений"""
        if len(values) < 2:
            return 'стабильный'
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Простая линейная регрессия
        try:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.01:
                return 'растущий'
            elif slope < -0.01:
                return 'падающий'
            else:
                return 'стабильный'
        except Exception as e:
            return 'неопределенный'
    
    def _generate_water_season_conclusion(self, target_column, key_dates, monthly_summary, season_analysis, year):
        """Сгенерировать вывод для сезона открытой воды"""
        
        conclusions = []
        
        if not key_dates:
            return ["Недостаточно данных для формирования вывода"]
        
        # Основная информация
        max_info = key_dates.get('max_value', {})
        min_info = key_dates.get('min_value', {})
        
        if max_info and min_info:
            conclusions.append(f"## 📊 Прогноз уровня воды на {year} год")
            conclusions.append("**Сезон открытой воды:** май - октябрь")
            conclusions.append("")
            conclusions.append(f"**📈 Максимальный уровень:** {max_info.get('value', 0):.2f} м ({max_info.get('formatted_date', 'N/A')})")
            conclusions.append(f"**📉 Минимальный уровень:** {min_info.get('value', 0):.2f} м ({min_info.get('formatted_date', 'N/A')})")
            conclusions.append(f"**📅 Период прогноза:** {key_dates.get('first_day', {}).get('formatted_date', 'N/A')} - {key_dates.get('last_day', {}).get('formatted_date', 'N/A')}")
            conclusions.append("")
        
        # Анализ по гидрологическим периодам
        if season_analysis:
            conclusions.append("### 🌊 Гидрологические периоды:")
            conclusions.append("")
            
            for period_key, period_data in season_analysis.items():
                conclusions.append(f"**{period_data['name']}:**")
                conclusions.append(f"- Средний уровень: {period_data['average']:.2f} м")
                conclusions.append(f"- Диапазон: от {period_data['min']:.2f} до {period_data['max']:.2f} м")
                conclusions.append(f"- Тренд: {period_data['trend']}")
                conclusions.append("")
        
        # Анализ по месяцам
        if monthly_summary:
            conclusions.append("### 📅 Анализ по месяцам:")
            conclusions.append("")
            
            for month_key, month_data in sorted(monthly_summary.items()):
                conclusions.append(f"**{month_data['month_name']}:**")
                conclusions.append(f"- Среднее: {month_data['average']:.2f} м")
                conclusions.append(f"- Минимум: {month_data['min']:.2f} м")
                conclusions.append(f"- Максимум: {month_data['max']:.2f} м")
                conclusions.append(f"- Стабильность: {'высокая' if month_data['std'] < month_data['average'] * 0.1 else 'средняя' if month_data['std'] < month_data['average'] * 0.2 else 'низкая'}")
                conclusions.append("")
        
        # Рекомендации по периодам
        conclusions.append("### 💡 Рекомендации по гидрологическим периодам:")
        conclusions.append("")
        
        if 'весенний_паводок' in season_analysis:
            spring_data = season_analysis['весенний_паводок']
            if spring_data['max'] > (spring_data['average'] * 1.3):
                conclusions.append("⚠️ **Весенний паводок (май):**")
                conclusions.append("- Ожидается высокий уровень паводка")
                conclusions.append("- Подготовьте защитные сооружения")
                conclusions.append("- Усильте мониторинг уровня воды")
            else:
                conclusions.append("✅ **Весенний паводок (май):**")
                conclusions.append("- Уровень в пределах нормы")
                conclusions.append("- Стандартный режим наблюдений")
            conclusions.append("")
        
        if 'летняя_межень' in season_analysis:
            summer_data = season_analysis['летняя_межень']
            if summer_data['min'] < (summer_data['average'] * 0.7):
                conclusions.append("⚠️ **Летняя межень (июнь-июль):**")
                conclusions.append("- Возможны низкие уровни воды")
                conclusions.append("- Планируйте водозаборы заранее")
                conclusions.append("- Мониторинг качества воды")
            else:
                conclusions.append("✅ **Летняя межень (июнь-июль):**")
                conclusions.append("- Стабильный уровень воды")
                conclusions.append("- Нормальный режим водопользования")
            conclusions.append("")
        
        if 'летние_дождевые_паводки' in season_analysis:
            flood_data = season_analysis['летние_дождевые_паводки']
            if flood_data['range'] > (flood_data['average'] * 0.5):
                conclusions.append("⚠️ **Дождевые паводки (август-сентябрь):**")
                conclusions.append("- Возможны резкие подъемы уровня")
                conclusions.append("- Будьте готовы к оперативному реагированию")
                conclusions.append("- Следите за прогнозом осадков")
            else:
                conclusions.append("✅ **Дождевые паводки (август-сентябрь):**")
                conclusions.append("- Стабильная гидрологическая обстановка")
                conclusions.append("- Низкий риск внезапных подъемов")
            conclusions.append("")
        
        if 'осенний_период' in season_analysis:
            autumn_data = season_analysis['осенний_период']
            if autumn_data['trend'] == 'растущий':
                conclusions.append("⚠️ **Осенний период (октябрь):**")
                conclusions.append("- Наблюдается рост уровня воды")
                conclusions.append("- Готовьтесь к завершению навигации")
                conclusions.append("- Проверьте зимние стоянки судов")
            else:
                conclusions.append("✅ **Осенний период (октябрь):**")
                conclusions.append("- Стабильный уровень воды")
                conclusions.append("- Подготовка к зимнему периоду")
            conclusions.append("")
        
        # Общие рекомендации
        conclusions.append("### 🎯 Общие рекомендации на сезон:")
        conclusions.append("")
        conclusions.append("1. **Подготовка к сезону (апрель):**")
        conclusions.append("   - Проверить защитные сооружения")
        conclusions.append("   - Подготовить оборудование для мониторинга")
        conclusions.append("   - Обновить планы действий при ЧС")
        conclusions.append("")
        conclusions.append("2. **Оперативный мониторинг (май-октябрь):**")
        conclusions.append("   - Ежедневный контроль уровня воды")
        conclusions.append("   - Особое внимание в паводковые периоды")
        conclusions.append("   - Своевременное оповещение при опасных уровнях")
        conclusions.append("")
        conclusions.append("3. **Завершение сезона (ноябрь):**")
        conclusions.append("   - Анализ данных за сезон")
        conclusions.append("   - Консервация оборудования")
        conclusions.append("   - Подготовка к зимнему периоду")
        conclusions.append("")
        conclusions.append("4. **Зимний период (ноябрь-апрель):**")
        conclusions.append("   - Мониторинг ледовой обстановки")
        conclusions.append("   - Плановое обслуживание оборудования")
        conclusions.append("   - Подготовка к следующему сезону")
        conclusions.append("")
        conclusions.append("---")
        conclusions.append("**Примечание:** Прогноз основан на исторических данных за сезон открытой воды.")
        conclusions.append("Зимние месяцы (ноябрь-апрель) не анализируются из-за ледового покрова.")
        
        return conclusions
    
    def _analyze_thresholds(self, forecast):
        """Анализ прогноза относительно порогов"""
        analysis = {
            'days_above_high': [],
            'days_above_danger': [],
            'max_exceedance_danger': 0,
            'max_exceedance_high': 0,
            'first_high_date': None,
            'first_danger_date': None,
            'last_high_date': None,
            'last_danger_date': None,
            'longest_high_period': 0,
            'danger_periods': []  # Список периодов опасного уровня
        }
        
        current_high_period = 0
        current_danger_period = 0
        danger_period_start = None
        has_high_exceeded = False
        has_danger_exceeded = False
        
        for pred in forecast['daily_predictions']:
            value = pred['value']
            date = pred['date']
            
            # Проверяем превышение опасного порога
            if value > self.danger_threshold:
                analysis['days_above_danger'].append(pred)
                exceedance = value - self.danger_threshold
                analysis['max_exceedance_danger'] = max(analysis['max_exceedance_danger'], exceedance)
                
                current_danger_period += 1
                if current_danger_period == 1:
                    danger_period_start = date
                
                if not has_danger_exceeded:
                    analysis['first_danger_date'] = date
                    has_danger_exceeded = True
                analysis['last_danger_date'] = date
            
            # Проверяем превышение высокого порога
            if value > self.high_threshold:
                analysis['days_above_high'].append(pred)
                exceedance = value - self.high_threshold
                analysis['max_exceedance_high'] = max(analysis['max_exceedance_high'], exceedance)
                
                current_high_period += 1
                
                if not has_high_exceeded:
                    analysis['first_high_date'] = date
                    has_high_exceeded = True
                analysis['last_high_date'] = date
            else:
                analysis['longest_high_period'] = max(analysis['longest_high_period'], current_high_period)
                current_high_period = 0
            
            # Завершаем период опасного уровня
            if value <= self.danger_threshold and current_danger_period > 0:
                if danger_period_start:
                    analysis['danger_periods'].append({
                        'start': danger_period_start,
                        'end': date - timedelta(days=1),
                        'duration': current_danger_period
                    })
                current_danger_period = 0
                danger_period_start = None
        
        # Проверяем последний период
        analysis['longest_high_period'] = max(analysis['longest_high_period'], current_high_period)
        
        # Завершаем последний период опасного уровня если он есть
        if current_danger_period > 0 and danger_period_start:
            analysis['danger_periods'].append({
                'start': danger_period_start,
                'end': forecast['daily_predictions'][-1]['date'],
                'duration': current_danger_period
            })
        
        # Рассчитываем продолжительность периодов
        if analysis['first_high_date'] and analysis['last_high_date']:
            duration = (analysis['last_high_date'] - analysis['first_high_date']).days + 1
            analysis['high_period_duration'] = duration
        else:
            analysis['high_period_duration'] = 0
            
        if analysis['first_danger_date'] and analysis['last_danger_date']:
            duration = (analysis['last_danger_date'] - analysis['first_danger_date']).days + 1
            analysis['danger_period_duration'] = duration
        else:
            analysis['danger_period_duration'] = 0
        
        # Суммарная продолжительность всех опасных периодов
        total_danger_days = sum([p['duration'] for p in analysis['danger_periods']])
        analysis['total_danger_days'] = total_danger_days
        
        return analysis
    
    def _classify_danger_days(self, forecast):
        """Классификация дней по уровню опасности"""
        classification = []
        
        for pred in forecast['daily_predictions']:
            value = pred['value']
            
            if value > self.danger_threshold:
                level = 'danger'
                color = 'red'
                icon = '🔴'
            elif value > self.high_threshold:
                level = 'warning'
                color = 'orange'
                icon = '🟡'
            else:
                level = 'normal'
                color = 'green'
                icon = '🟢'
            
            classification.append({
                'date': pred['date'],
                'formatted_date': pred['formatted_date'],
                'value': value,
                'level': level,
                'color': color,
                'icon': icon,
                'exceedance_high': value - self.high_threshold if value > self.high_threshold else None,
                'exceedance_danger': value - self.danger_threshold if value > self.danger_threshold else None,
                'month': pred['month'],
                'month_name': pred['month_name']
            })
        
        return classification
    
    def generate_threshold_recommendations(self, analysis):
        """Генерация рекомендаций на основе анализа"""
        recommendations = []
        
        # Количество дней с превышением
        high_days = len(analysis.get('days_above_high', []))
        danger_days = analysis.get('total_danger_days', 0)
        danger_periods = analysis.get('danger_periods', [])
        
        if danger_days > 0:
            max_exceed = analysis.get('max_exceedance_danger', 0)
            
            if max_exceed > self.danger_threshold * 0.2:  # Более 20% превышения
                recommendations.append({
                    'type': 'critical',
                    'text': "⚠️ **КРИТИЧЕСКАЯ СИТУАЦИЯ**",
                    'details': [
                        f"Ожидается {danger_days} дней с превышением опасного уровня",
                        f"Максимальное превышение: {max_exceed:.2f} м",
                        f"Количество опасных периодов: {len(danger_periods)}"
                    ],
                    'actions': [
                        "Активировать систему оповещения населения",
                        "Подготовить аварийные бригады",
                        "Рассмотреть возможность эвакуации из низменных районов",
                        "Усилить круглосуточный мониторинг уровня воды"
                    ]
                })
            else:
                recommendations.append({
                    'type': 'warning',
                    'text': "⚠️ **ВЫСОКАЯ ОПАСНОСТЬ**",
                    'details': [
                        f"{danger_days} дней выше опасного уровня",
                        f"Количество периодов: {len(danger_periods)}"
                    ],
                    'actions': [
                        "Усилить мониторинг уровня воды",
                        "Подготовить защитные сооружения",
                        "Оповестить ответственные службы",
                        "Проверить готовность насосного оборудования"
                    ]
                })
            
            # Добавляем информацию о периодах
            if danger_periods:
                period_details = []
                for i, period in enumerate(danger_periods[:3]):  # Показываем первые 3 периода
                    start_str = period['start'].strftime('%d.%m')
                    end_str = period['end'].strftime('%d.%m')
                    period_details.append(f"Период {i+1}: {start_str}-{end_str} ({period['duration']} дн.)")
                
                if period_details:
                    recommendations[-1]['periods'] = period_details
        
        elif high_days > 0:
            duration = analysis.get('high_period_duration', 0)
            longest_period = analysis.get('longest_high_period', 0)
            
            if longest_period > 14:
                recommendations.append({
                    'type': 'warning',
                    'text': "🟡 **ДЛИТЕЛЬНЫЙ ПЕРИОД ПОВЫШЕННОГО УРОВНЯ**",
                    'details': [
                        f"{high_days} дней выше высокого уровня",
                        f"Наиболее длительный непрерывный период: {longest_period} дней",
                        f"Общая продолжительность: {duration} дней"
                    ],
                    'actions': [
                        "Усилить наблюдение за гидротехническими сооружениями",
                        "Проверить систему водоотведения",
                        "Обновлять прогнозы ежедневно",
                        "Информировать водопользователей"
                    ]
                })
            else:
                recommendations.append({
                    'type': 'info',
                    'text': "🟡 **ПОВЫШЕННЫЙ УРОВЕНЬ ВОДЫ**",
                    'details': [
                        f"{high_days} дней выше высокого уровня",
                        f"Общая продолжительность: {duration} дней"
                    ],
                    'actions': [
                        "Мониторить изменение уровня воды",
                        "Проверить готовность оборудования",
                        "Информировать ответственных лиц",
                        "Следить за прогнозом осадков"
                    ]
                })
        else:
            recommendations.append({
                'type': 'success',
                'text': "✅ **НОРМАЛЬНАЯ ГИДРОЛОГИЧЕСКАЯ ОБСТАНОВКА**",
                'details': [
                    "Превышений порогов не ожидается",
                    "Уровень воды в пределах нормы"
                ],
                'actions': [
                    "Продолжить регулярный мониторинг",
                    "Обновлять прогноз раз в неделю",
                    "Вести журнал наблюдений",
                    "Подготовиться к следующему паводковому периоду"
                ]
            })
        
        # Добавляем рекомендации по сезону
        recommendations.append({
            'type': 'info',
            'text': "📅 **РЕКОМЕНДАЦИИ ПО СЕЗОНУ ОТКРЫТОЙ ВОДЫ:**",
            'details': [
                "Сезон анализа: май - октябрь",
                "Зимние месяцы (ноябрь-апрель) не анализируются",
                "Модель обучена только на данных периода открытой воды"
            ],
            'actions': [
                "Планируйте мероприятия с учетом сезонности",
                "Учитывайте, что зимой река замерзает",
                "Подготовьтесь к началу сезона заранее"
            ]
        })
        
        return recommendations
    
    def get_final_conclusion(self, forecasts, date_column):
        """Получить итоговый вывод по всем прогнозам"""
        
        conclusions = []
        
        if not forecasts:
            return ["Нет данных для формирования вывода"]
        
        conclusions.append("## 📋 ИТОГОВЫЙ ОТЧЕТ ПО СЕЗОНУ ОТКРЫТОЙ ВОДЫ")
        conclusions.append("")
        
        # Анализируем каждый прогноз
        for target, forecast in forecasts.items():
            if 'conclusion' in forecast and forecast['conclusion']:
                conclusions.append(f"### 📊 {target}")
                for line in forecast['conclusion']:
                    if isinstance(line, str) and line.strip():
                        conclusions.append(line)
                conclusions.append("")
        
        # Общие выводы
        conclusions.append("### 💡 КЛЮЧЕВЫЕ ВЫВОДЫ:")
        
        # Проверяем наличие экстремальных значений
        extreme_events = []
        for target, forecast in forecasts.items():
            key_dates = forecast.get('key_dates', {})
            if 'max_value' in key_dates:
                max_val = key_dates['max_value'].get('value', 0)
                max_date = key_dates['max_value'].get('formatted_date', 'N/A')
                
                if 'температур' in target.lower() and max_val > 25:
                    extreme_events.append(f"🌡️ **Жара:** {max_val:.1f}°C ({max_date})")
                elif 'осадк' in target.lower() and max_val > 20:
                    extreme_events.append(f"🌧️ **Сильные осадки:** {max_val:.1f} мм ({max_date})")
                elif 'уровен' in target.lower() and max_val > 5:
                    extreme_events.append(f"🌊 **Высокий уровень:** {max_val:.1f} м ({max_date})")
        
        if extreme_events:
            conclusions.append("⚠️ **ОЖИДАЮТСЯ ЭКСТРЕМАЛЬНЫЕ СОБЫТИЯ:**")
            for event in extreme_events:
                conclusions.append(f"- {event}")
        else:
            conclusions.append("✅ **ЭКСТРЕМАЛЬНЫЕ СОБЫТИЯ НЕ ОЖИДАЮТСЯ**")
        
        conclusions.append("")
        conclusions.append("### 🎯 РЕКОМЕНДАЦИИ К ДЕЙСТВИЮ:")
        
        # Рекомендации в зависимости от прогнозов
        has_high_temp = False
        has_high_precip = False
        has_high_water = False
        
        for target, forecast in forecasts.items():
            key_dates = forecast.get('key_dates', {})
            if 'max_value' in key_dates:
                max_val = key_dates['max_value'].get('value', 0)
                
                if 'температур' in target.lower() and max_val > 25:
                    has_high_temp = True
                elif 'осадк' in target.lower() and max_val > 20:
                    has_high_precip = True
                elif 'уровен' in target.lower() and max_val > 5:
                    has_high_water = True
        
        if has_high_temp:
            conclusions.append("1. **Подготовьтесь к жаре:**")
            conclusions.append("   - Проверьте системы кондиционирования")
            conclusions.append("   - Обеспечьте достаточный запас воды")
            conclusions.append("   - Избегайте длительного пребывания на солнце")
        
        if has_high_precip:
            conclusions.append("2. **Подготовьтесь к осадкам:**")
            conclusions.append("   - Проверьте дренажные системы")
            conclusions.append("   - Подготовьте средства защиты от дождя")
            conclusions.append("   - Будьте готовы к возможным паводкам")
        
        if has_high_water:
            conclusions.append("3. **Мониторинг уровня воды:**")
            conclusions.append("   - Регулярно проверяйте уровень воды")
            conclusions.append("   - Подготовьте план эвакуации при необходимости")
            conclusions.append("   - Проверьте защитные сооружения")
        
        if not (has_high_temp or has_high_precip or has_high_water):
            conclusions.append("1. **Стандартные рекомендации:**")
            conclusions.append("   - Продолжайте регулярные наблюдения")
            conclusions.append("   - Сравнивайте прогноз с фактическими данными")
            conclusions.append("   - Обновляйте данные для повышения точности прогнозов")
        
        conclusions.append("")
        conclusions.append("### 📅 СЕЗОННЫЕ ОСОБЕННОСТИ:")
        conclusions.append("- **Период анализа:** май - октябрь (сезон открытой воды)")
        conclusions.append("- **Зимний период:** ноябрь - апрель (ледовый покров, данные отсутствуют)")
        conclusions.append("- **Рекомендуемые сроки подготовки:** апрель (до начала сезона)")
        
        conclusions.append("")
        conclusions.append("---")
        conclusions.append("**Срок действия прогноза:** 1 сезон (май-октябрь)")
        conclusions.append("**Точность:** Высокая на основе исторических данных сезона")
        conclusions.append("**Важно:** Прогнозы носят вероятностный характер и основаны на данных только за сезон открытой воды")
        
        return conclusions