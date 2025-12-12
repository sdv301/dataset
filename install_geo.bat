@echo off
echo ========================================
echo Установка геоаналитического анализатора
echo ========================================

echo 1. Установка основных библиотек...
pip install streamlit pandas numpy plotly

echo 2. Установка гео-библиотек...
pip install geopandas shapely folium streamlit-folium

echo 3. Установка визуализации...
pip install matplotlib seaborn

echo 4. Установка обработки данных...
pip install openpyxl scikit-learn scipy

echo.
echo ✅ Установка завершена!
echo.
echo Для запуска выполните:
echo streamlit run app.py
pause