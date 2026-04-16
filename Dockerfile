# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем систему и устанавливаем системные зависимости для сборки C++ библиотек (XGBoost/Pandas/GeoPandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask

# Копируем исходный код проекта
COPY . .

# Supervisor конфигурация для запуска обоих серверов
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Открываем порты для Streamlit и Flask API
EXPOSE 8501 8502

# Команда для запуска supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
