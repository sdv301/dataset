import pandas as pd
import numpy as np
from datetime import datetime
import json

def format_number(num):
    """Форматировать число для отображения"""
    if isinstance(num, (int, np.integer)):
        return f"{num:,}".replace(",", " ")
    elif isinstance(num, (float, np.float)):
        return f"{num:,.2f}".replace(",", " ").replace(".", ",")
    return str(num)

def get_timestamp():
    """Получить текущую метку времени"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_dataframe(df):
    """Проверить валидность DataFrame"""
    if df is None:
        return False, "DataFrame is None"
    
    if len(df) == 0:
        return False, "DataFrame is empty"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    return True, "Valid DataFrame"

def convert_to_serializable(obj):
    """Конвертировать объект в сериализуемый формат"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return str(obj)