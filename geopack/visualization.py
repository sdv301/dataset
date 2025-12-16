import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
from datetime import datetime
from pathlib import Path
import seaborn as sns
from typing import Optional, Union, List, Dict, Any
import json

class Visualizer:
    def __init__(self, theme: str = 'plotly_white'):
        """
        Инициализация визуализатора
        
        Args:
            theme: Тема оформления графиков ('plotly', 'plotly_white', 'plotly_dark', 
                   'ggplot2', 'seaborn', 'simple_white', 'none')
        """
        self.theme = theme
        self.color_palettes = {
            # Sequential color scales
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma,
            'cividis': px.colors.sequential.Cividis,
            'thermal': px.colors.sequential.thermal,
            'ice': px.colors.sequential.ice,
            'emrld': px.colors.sequential.Emrld,
            'deep': px.colors.sequential.deep,
            'dense': px.colors.sequential.dense,
            'gray': px.colors.sequential.gray,
            'purples': px.colors.sequential.Purples,
            'blues': px.colors.sequential.Blues,
            'greens': px.colors.sequential.Greens,
            'oranges': px.colors.sequential.Oranges,
            'reds': px.colors.sequential.Reds,
            
            # Diverging color scales
            'rdbu': px.colors.diverging.RdBu,
            'rdylbu': px.colors.diverging.RdYlBu,
            'spectral': px.colors.diverging.Spectral,
            'balance': px.colors.diverging.balance,
            'delta': px.colors.diverging.delta,
            
            # Cyclic color scales
            'twilight': px.colors.cyclical.Twilight,
            'icefire': px.colors.cyclical.IceFire,
            'phase': px.colors.cyclical.Phase,
            
            # Qualitative color scales
            'set1': px.colors.qualitative.Set1,
            'set2': px.colors.qualitative.Set2,
            'set3': px.colors.qualitative.Set3,
            'pastel1': px.colors.qualitative.Pastel1,
            'pastel2': px.colors.qualitative.Pastel2,
            'dark2': px.colors.qualitative.Dark2,
            'bold': px.colors.qualitative.Bold,
            'safe': px.colors.qualitative.Safe,
            'vivid': px.colors.qualitative.Vivid,
            'alphabet': px.colors.qualitative.Alphabet,
        }
        
        self.default_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 
                                   'drawcircle', 'drawrect', 'eraseshape'],
            'scrollZoom': True,
            'responsive': True
        }
    
    # ==================== ОСНОВНЫЕ ГРАФИКИ ====================
    
    def create_scatter_plot(self, df, x_col, y_col, color_col=None, 
                           size_col=None, hover_cols=None, 
                           title="Точечный график", trendline=None,
                           marginal_x=None, marginal_y=None,
                           facet_col=None, facet_row=None,
                           height=600, width=None, color_scale=None):
        """Создать улучшенный точечный график с множеством опций"""
        hover_data = hover_cols if hover_cols else df.columns.tolist()
        
        if color_scale is None:
            color_scale = self.color_palettes.get('viridis')
        
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col, size=size_col,
            hover_data=hover_data, title=title,
            trendline=trendline, marginal_x=marginal_x, marginal_y=marginal_y,
            facet_col=facet_col, facet_row=facet_row,
            height=height, width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get('set3'),
            color_continuous_scale=color_scale
        )
        
        fig.update_layout(
            title_font_size=16,
            hoverlabel=dict(bgcolor="white", font_size=12),
            legend=dict(title=color_col if color_col else None, orientation="h", y=-0.2)
        )
        
        if color_col and pd.api.types.is_numeric_dtype(df[color_col]):
            fig.update_coloraxes(colorbar_title=color_col)
        
        return fig
    
    def create_line_plot(self, df, x_col, y_cols, color_col=None,
                        title="Линейный график", mode='lines+markers',
                        line_shape='linear', animation_frame=None,
                        height=500, width=None, color_palette='set1'):
        """Создать многофункциональный линейный график"""
        if isinstance(y_cols, str):
            y_cols = [y_cols]
        
        fig = go.Figure()
        
        colors = self.color_palettes.get(color_palette, self.color_palettes['set1'])
        
        for i, y_col in enumerate(y_cols):
            if color_col:
                categories = df[color_col].unique()
                for j, category in enumerate(categories):
                    subset = df[df[color_col] == category]
                    fig.add_trace(go.Scatter(
                        x=subset[x_col], y=subset[y_col],
                        mode=mode,
                        name=f"{y_col} - {category}",
                        line_shape=line_shape,
                        line=dict(width=2, color=colors[j % len(colors)])
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[y_col],
                    mode=mode,
                    name=y_col,
                    line_shape=line_shape,
                    line=dict(width=2, color=colors[i % len(colors)])
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title=x_col,
            yaxis_title="Значение",
            hovermode='x unified',
            template=self.theme,
            height=height,
            width=width,
            legend=dict(orientation="h", y=-0.2)
        )
        
        return fig
    
    def create_histogram(self, df, x_col, color_col=None, 
                        title="Гистограмма", nbins=None,
                        cumulative=False, histnorm=None,
                        barmode='overlay', opacity=0.7,
                        height=500, width=None, color_palette='pastel1'):
        """Создать улучшенную гистограмму"""
        fig = px.histogram(
            df, x=x_col, color=color_col,
            title=title, nbins=nbins,
            cumulative=cumulative, histnorm=histnorm,
            barmode=barmode, opacity=opacity,
            height=height, width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_layout(
            bargap=0.1,
            title_font_size=16,
            xaxis_title=x_col,
            yaxis_title="Количество" if not histnorm else "Плотность"
        )
        
        return fig
    
    def create_density_plot(self, df, columns, title="График плотности", 
                           height=500, width=None, color_palette='set2'):
        """Создать график плотности распределения"""
        fig = go.Figure()
        
        colors = self.color_palettes.get(color_palette, self.color_palettes['set2'])
        
        for i, col in enumerate(columns):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                data = df[col].dropna()
                if len(data) > 1:
                    kde = sns.kdeplot(data)
                    x, y = kde.get_lines()[0].get_data()
                    plt.close()  # Закрываем matplotlib figure
                    
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=2),
                        fill='tozeroy',
                        opacity=0.3
                    ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="Значение",
            yaxis_title="Плотность",
            template=self.theme,
            height=height,
            width=width,
            hovermode='x unified'
        )
        
        return fig
    
    def create_bar_chart(self, df, x_col, y_col, color_col=None,
                        title="Столбчатая диаграмма", barmode='relative',
                        text_auto=False, orientation='v',
                        height=500, width=None, color_palette='bold'):
        """Создать улучшенную столбчатую диаграмму"""
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            title=title, barmode=barmode,
            text_auto=text_auto, orientation=orientation,
            height=height, width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title=x_col if orientation == 'v' else y_col,
            yaxis_title=y_col if orientation == 'v' else x_col,
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        
        if orientation == 'h':
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig
    
    def create_stacked_bar_chart(self, df, x_col, y_cols, 
                                title="Сложенная столбчатая диаграмма",
                                height=500, width=None, color_palette='set2'):
        """Создать сложенную столбчатую диаграмму"""
        fig = go.Figure()
        
        colors = self.color_palettes.get(color_palette, self.color_palettes['set2'])
        
        for i, y_col in enumerate(y_cols):
            fig.add_trace(go.Bar(
                x=df[x_col],
                y=df[y_col],
                name=y_col,
                marker_color=colors[i % len(colors)],
                opacity=0.8
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title=x_col,
            yaxis_title="Значение",
            barmode='stack',
            template=self.theme,
            height=height,
            width=width,
            legend=dict(orientation="h", y=-0.2)
        )
        
        return fig
    
    def create_pie_chart(self, df, column, title="Круговая диаграмма", 
                        limit=10, hole=0.3, pull=None,
                        height=500, width=None, color_palette='pastel1'):
        """Создать улучшенную круговую диаграмму"""
        if df[column].nunique() > limit:
            values = df[column].value_counts().head(limit)
            others_count = df[column].value_counts().iloc[limit:].sum()
            if others_count > 0:
                values = pd.concat([values, pd.Series([others_count], index=['Другие'])])
        else:
            values = df[column].value_counts()
        
        if pull is None:
            pull = [0.1] * len(values)
        
        fig = px.pie(
            values=values.values,
            names=values.index,
            title=title,
            hole=hole,
            height=height,
            width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=pull,
            marker=dict(line=dict(color='white', width=2))
        )
        
        fig.update_layout(
            title_font_size=16,
            showlegend=True,
            legend=dict(orientation="h", y=-0.1)
        )
        
        return fig
    
    def create_box_plot(self, df, x_col, y_col, color_col=None,
                       title="Ящик с усами", points='outliers',
                       notched=False, height=500, width=None, color_palette='set1'):
        """Создать улучшенный box plot"""
        fig = px.box(
            df, x=x_col, y=y_col, color=color_col,
            title=title, points=points,
            notched=notched,
            height=height, width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title=x_col,
            yaxis_title=y_col,
            boxmode='group'
        )
        
        return fig
    
    def create_violin_plot(self, df, x_col, y_col, color_col=None,
                          title="Скрипичная диаграмма", box=False,
                          height=500, width=None, color_palette='pastel1'):
        """Создать скрипичную диаграмму"""
        fig = px.violin(
            df, x=x_col, y=y_col, color=color_col,
            title=title, box=box,
            height=height, width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    # ==================== МНОГОМЕРНЫЕ ГРАФИКИ ====================
    
    def create_correlation_matrix(self, df, title="Корреляционная матрица",
                                method='pearson', height=600, width=800,
                                color_scale='rdbu'):
        """Создать улучшенную корреляционную матрицу"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        corr = df[numeric_cols].corr(method=method)
        
        colorscale = self.color_palettes.get(color_scale, 'RdBu')
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Корреляция")  
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(tickangle=45),
            height=height,
            width=width,
            template=self.theme
        )
        
        return fig
    
    def create_scatter_matrix(self, df, dimensions=None, color_col=None,
                             title="Матрица рассеяния", height=800, width=800,
                             color_palette='set1'):
        """Создать матрицу рассеяния"""
        if dimensions is None:
            dimensions = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(dimensions) > 6:
                dimensions = dimensions[:6]
        
        fig = px.scatter_matrix(
            df,
            dimensions=dimensions,
            color=color_col,
            title=title,
            height=height,
            width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(title_font_size=18)
        
        return fig
    
    def create_parallel_coordinates(self, df, color_col, 
                                   title="Параллельные координаты",
                                   height=500, width=800,
                                   color_scale='phase'):
        """Создать диаграмму параллельных координат"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        colorscale = self.color_palettes.get(color_scale, px.colors.diverging.Tealrose)
        
        fig = px.parallel_coordinates(
            df,
            color=color_col,
            dimensions=numeric_cols,
            title=title,
            height=height,
            width=width,
            template=self.theme,
            color_continuous_scale=colorscale
        )
        
        fig.update_layout(title_font_size=16)
        
        return fig
    
    def create_3d_scatter(self, df, x_col, y_col, z_col, color_col=None,
                         size_col=None, title="3D Scatter Plot",
                         height=700, width=800, color_scale='viridis'):
        """Создать 3D точечный график"""
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col,
            color=color_col, size=size_col,
            title=title,
            height=height,
            width=width,
            template=self.theme,
            color_continuous_scale=self.color_palettes.get(color_scale)
        )
        
        fig.update_layout(
            title_font_size=16,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return fig
    
    # ==================== ВРЕМЕННЫЕ РЯДЫ ====================
    
    def create_time_series(self, df, time_col, value_cols,
                          title="Временной ряд", height=500, width=None,
                          color_palette='bold'):
        """Создать график временного ряда"""
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        
        fig = go.Figure()
        
        colors = self.color_palettes.get(color_palette, self.color_palettes['bold'])
        
        for i, value_col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[value_col],
                mode='lines',
                name=value_col,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="Время",
            yaxis_title="Значение",
            hovermode='x unified',
            template=self.theme,
            height=height,
            width=width,
            legend=dict(orientation="h", y=-0.2)
        )
        
        return fig
    
    def create_calendar_heatmap(self, df, date_col, value_col,
                               title="Календарная тепловая карта",
                               year=None, height=300, width=800,
                               color_scale='thermal'):
        """Создать календарную тепловую карту"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['weekday'] = df[date_col].dt.weekday
        df['week'] = df[date_col].dt.isocalendar().week
        
        if year:
            df = df[df['year'] == year]
        
        pivot = df.pivot_table(
            values=value_col,
            index='month',
            columns='day',
            aggfunc='mean'
        ).fillna(0)
        
        colorscale = self.color_palettes.get(color_scale, 'Viridis')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            text=pivot.round(2).values,
            texttemplate='%{text}',
            hoverongaps=False,
            colorbar=dict(title=value_col)
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="День",
            yaxis_title="Месяц",
            height=height,
            width=width,
            template=self.theme
        )
        
        return fig
    
    # ==================== ГЕОПРОСТРАНСТВЕННАЯ ВИЗУАЛИЗАЦИЯ ====================
    
    def create_base_map(self, center_lat=None, center_lon=None, 
                       gdf=None, tiles="OpenStreetMap", zoom_start=10,
                       height=500, width=800, attribution=None):
        """Создать улучшенную базовую карту"""
        if gdf is not None and not gdf.empty:
            center_lat = gdf.geometry.y.mean()
            center_lon = gdf.geometry.x.mean()
        elif center_lat is None or center_lon is None:
            center_lat, center_lon = 55.7558, 37.6173  # Москва по умолчанию
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tiles,
            control_scale=True,
            height=f"{height}px",
            width=f"{width}px",
            attr=attribution
        )
        
        # Добавляем полезные плагины
        Fullscreen().add_to(m)
        
        return m
    
    def add_markers_to_map(self, m, gdf, df=None, color_by=None,
                          size_by=None, popup_cols=None,
                          cluster=False, radius=8, opacity=0.7,
                          color_palette='set1'):
        """Добавить маркеры на карту с улучшенными опциями"""
        if df is None:
            df = gdf.copy()
        
        if popup_cols is None:
            popup_cols = df.columns.tolist()[:5]
        
        if cluster:
            marker_cluster = MarkerCluster().add_to(m)
            map_to_add = marker_cluster
        else:
            map_to_add = m
        
        colors = self.color_palettes.get(color_palette, self.color_palettes['set1'])
        
        for idx, row in gdf.iterrows():
            popup_text = f"<b>Объект {idx}</b><br>"
            for col in popup_cols:
                if col in df.columns:
                    popup_text += f"<b>{col}:</b> {row[col]}<br>"
            
            if color_by and color_by in df.columns:
                color_val = df.loc[idx, color_by]
                if pd.api.types.is_numeric_dtype(df[color_by]):
                    norm_val = (color_val - df[color_by].min()) / (df[color_by].max() - df[color_by].min())
                    color = plt.cm.viridis(norm_val)
                    color_hex = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                else:
                    # Для категориальных данных
                    unique_vals = df[color_by].unique().tolist()
                    color_index = unique_vals.index(color_val) if color_val in unique_vals else 0
                    color_hex = colors[color_index % len(colors)]
            else:
                color_hex = '#3388ff'
            
            if size_by and size_by in df.columns:
                size_val = df.loc[idx, size_by]
                if pd.api.types.is_numeric_dtype(df[size_by]):
                    norm_size = (size_val - df[size_by].min()) / (df[size_by].max() - df[size_by].min())
                    radius = 5 + norm_size * 15
                else:
                    radius = 8
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color_hex,
                fill=True,
                fill_color=color_hex,
                fill_opacity=opacity,
                weight=1
            ).add_to(map_to_add)
        
        return m
    
    def create_choropleth_map(self, gdf, value_column, 
                             title="Хороплетная карта",
                             legend_name="Значение",
                             height=600, width=800,
                             fill_color='YlOrRd'):
        """Создать хороплетную карту"""
        m = self.create_base_map(gdf=gdf, height=height, width=width)
        
        # Создаем уникальный идентификатор для каждой строки
        gdf = gdf.copy()
        gdf['feature_id'] = range(len(gdf))
        
        folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            data=gdf,
            columns=['feature_id', value_column],
            key_on='feature.id',
            fill_color=fill_color,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=legend_name,
            highlight=True,
            nan_fill_color="White"
        ).add_to(m)
        
        # Добавляем всплывающие подсказки
        style_function = lambda x: {'fillColor': '#ffffff', 
                                   'color':'#000000', 
                                   'fillOpacity': 0.1, 
                                   'weight': 0.1}
        highlight_function = lambda x: {'fillColor': '#000000', 
                                       'color':'#000000', 
                                       'fillOpacity': 0.50, 
                                       'weight': 0.1}
        
        for idx, row in gdf.iterrows():
            popup_text = f"<b>Район {idx}</b><br><b>{value_column}:</b> {row[value_column]}"
            
            folium.GeoJson(
                row.geometry,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.Tooltip(popup_text, sticky=False)
            ).add_to(m)
        
        m.get_root().html.add_child(folium.Element(f"<h3 align='center'>{title}</h3>"))
        
        return m
    
    def create_heatmap(self, gdf, value_column=None, radius=15, blur=25,
                      min_opacity=0.4, max_zoom=10, height=600, width=800):
        """Создать улучшенную тепловую карту"""
        m = self.create_base_map(gdf=gdf, height=height, width=width)
        
        if value_column and value_column in gdf.columns:
            heat_data = [[row.geometry.y, row.geometry.x, row[value_column]] 
                        for _, row in gdf.iterrows()]
        else:
            heat_data = [[row.geometry.y, row.geometry.x, 1] for _, row in gdf.iterrows()]
        
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            max_zoom=max_zoom,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)
        
        return m
    
    def create_voronoi_diagram(self, gdf, buffer_percent=0.1,
                              height=600, width=800):
        """Создать улучшенную диаграмму Вороного"""
        try:
            points = np.array([(geom.x, geom.y) for geom in gdf.geometry])
            
            # Добавляем буферные точки для лучшего отображения
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            x_buffer = (x_max - x_min) * buffer_percent
            y_buffer = (y_max - y_min) * buffer_percent
            
            buffer_points = np.array([
                [x_min - x_buffer, y_min - y_buffer],
                [x_min - x_buffer, y_max + y_buffer],
                [x_max + x_buffer, y_min - y_buffer],
                [x_max + x_buffer, y_max + y_buffer]
            ])
            
            all_points = np.vstack([points, buffer_points])
            vor = Voronoi(all_points)
            
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Рисуем диаграмму Вороного
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, 
                           line_colors='orange', line_width=2, 
                           line_alpha=0.6, point_size=0)
            
            # Рисуем исходные точки
            ax.scatter(points[:, 0], points[:, 1], c='red', s=50, 
                      alpha=0.7, edgecolors='black', linewidth=1)
            
            # Настраиваем отображение
            ax.set_title("Диаграмма Вороного", fontsize=16, fontweight='bold')
            ax.set_xlabel("Долгота", fontsize=12)
            ax.set_ylabel("Широта", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            import streamlit as st
            st.warning(f"Не удалось создать диаграмму Вороного: {e}")
            return None
    
    # ==================== СПЕЦИАЛЬНЫЕ ВИЗУАЛИЗАЦИИ ====================
    
    def create_gauge_chart(self, value, min_val, max_val, 
                          title="Индикатор", height=300, width=400):
        """Создать индикатор (gauge chart)"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, min_val + (max_val-min_val)/3], 'color': "lightgray"},
                    {'range': [min_val + (max_val-min_val)/3, min_val + 2*(max_val-min_val)/3], 'color': "gray"},
                    {'range': [min_val + 2*(max_val-min_val)/3, max_val], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=height,
            width=width,
            template=self.theme
        )
        
        return fig
    
    def create_radar_chart(self, df, categories, values,
                          title="Радарная диаграмма", height=500, width=500,
                          color_palette='set1'):
        """Создать радарную диаграмма"""
        fig = go.Figure()
        
        colors = self.color_palettes.get(color_palette, self.color_palettes['set1'])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name="Показатели",
            line_color=colors[0] if colors else 'blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )
            ),
            showlegend=False,
            title=dict(text=title, font=dict(size=18)),
            height=height,
            width=width,
            template=self.theme
        )
        
        return fig
    
    def create_treemap(self, df, path_cols, value_col,
                      title="Древовидная карта", height=600, width=800,
                      color_palette='pastel1'):
        """Создать древовидную карту"""
        fig = px.treemap(
            df,
            path=path_cols,
            values=value_col,
            title=title,
            height=height,
            width=width,
            template=self.theme,
            color_discrete_sequence=self.color_palettes.get(color_palette)
        )
        
        fig.update_layout(
            title_font_size=16,
            margin=dict(t=50, l=25, r=25, b=25)
        )
        
        fig.update_traces(
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>Значение: %{value}<br>%{percentParent} от родителя'
        )
        
        return fig
    
    # ==================== КОМБИНИРОВАННЫЕ ГРАФИКИ ====================
    
    def create_subplots(self, plots, rows=1, cols=2, 
                       shared_xaxes=False, shared_yaxes=False,
                       vertical_spacing=0.1, horizontal_spacing=0.1,
                       subplot_titles=None, height=400, width=800):
        """Создать комбинированный график с несколькими панелями"""
        fig = make_subplots(
            rows=rows, cols=cols,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            subplot_titles=subplot_titles
        )
        
        for i, plot_func in enumerate(plots):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            subplot_fig = plot_func()
            for trace in subplot_fig.data:
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            height=height,
            width=width,
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    # ==================== УТИЛИТЫ ====================
    
    def get_available_color_palettes(self):
        """Получить список доступных цветовых палитр"""
        return list(self.color_palettes.keys())
    
    def get_plot_types(self):
        """Получить список доступных типов графиков"""
        plot_types = {
            'Основные': [
                'scatter', 'line', 'bar', 'histogram', 'pie',
                'box', 'violin', 'density'
            ],
            'Многомерные': [
                'scatter_matrix', 'parallel_coordinates',
                'correlation_matrix', '3d_scatter'
            ],
            'Временные ряды': [
                'time_series', 'calendar_heatmap'
            ],
            'Геопространственные': [
                'base_map', 'choropleth', 'heatmap',
                'voronoi', 'markers_map'
            ],
            'Специальные': [
                'gauge', 'radar', 'treemap', 'stacked_bar'
            ]
        }
        return plot_types
    
    def auto_visualize(self, df, target_column=None, max_columns=10):
        """Автоматически создать подходящие визуализации для данных"""
        visualizations = []
        
        # Ограничиваем количество колонок для анализа
        analysis_cols = df.select_dtypes(include=[np.number, 'datetime']).columns.tolist()
        if len(analysis_cols) > max_columns:
            analysis_cols = analysis_cols[:max_columns]
        
        # Для числовых данных
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            # Матрица корреляции
            corr_fig = self.create_correlation_matrix(df[numeric_cols])
            if corr_fig:
                visualizations.append(corr_fig)
            
            # Гистограммы для каждой числовой колонки
            for col in numeric_cols[:5]:
                visualizations.append(self.create_histogram(df, col, title=f"Распределение {col}"))
        
        # Для категориальных данных
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols[:3]:
            if df[col].nunique() < 20:
                # Создаем DataFrame с подсчетом значений
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'count']
                
                visualizations.append(self.create_bar_chart(
                    value_counts, col, 'count', 
                    title=f"Распределение {col}"
                ))
        
        # Если есть целевая переменная
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # Для числовой целевой переменной
                for col in numeric_cols[:3]:
                    if col != target_column:
                        visualizations.append(self.create_scatter_plot(
                            df, col, target_column,
                            title=f"{col} vs {target_column}"
                        ))
            else:
                # Для категориальной целевой переменной
                for col in numeric_cols[:3]:
                    visualizations.append(self.create_box_plot(
                        df, target_column, col,
                        title=f"{col} по {target_column}"
                    ))
        
        return [v for v in visualizations if v is not None]
    
    def save_plot(self, fig, filename, folder="visualizations",
                 formats=None, dpi=300):
        """Сохранить график в нескольких форматах"""
        if formats is None:
            formats = ['html', 'png']
        
        saved_files = []
        
        for fmt in formats:
            try:
                if fmt.lower() == 'html':
                    output_path = Path("reports") / folder / f"{filename}.html"
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    if hasattr(fig, 'to_html'):
                        html_str = fig.to_html(include_plotlyjs='cdn')
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(html_str)
                        saved_files.append(output_path)
                    elif hasattr(fig, 'save'):
                        fig.save(output_path)
                        saved_files.append(output_path)
                
                elif fmt.lower() in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
                    output_path = Path("reports") / folder / f"{filename}.{fmt}"
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    if hasattr(fig, 'write_image'):
                        fig.write_image(str(output_path), format=fmt, scale=2)
                        saved_files.append(output_path)
                    elif hasattr(fig, 'savefig'):
                        fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
                        saved_files.append(output_path)
                
                elif fmt.lower() == 'json':
                    output_path = Path("reports") / folder / f"{filename}.json"
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    if hasattr(fig, 'to_json'):
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(fig.to_json())
                        saved_files.append(output_path)
            
            except Exception as e:
                import streamlit as st
                st.warning(f"Не удалось сохранить график в формате {fmt}: {e}")
        
        return saved_files