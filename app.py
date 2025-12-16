import streamlit as st
from geopack.ui_components import setup_app, render_sidebar, render_main_content, render_footer

def main():
    """Главная функция приложения"""
    # Настройка приложения
    setup_app()
    
    # Рендер сайдбара
    render_sidebar()
    
    # Основной контент
    render_main_content()
    
    # Футер
    render_footer()

if __name__ == "__main__":
    main()