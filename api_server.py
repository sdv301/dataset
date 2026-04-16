#!/usr/bin/env python3
"""
Flask API для прогноза паводков
"""
import os
import json
from flask import Flask, jsonify

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def _load_water_data():
    """Загружает данные из файлов в папке data/ если они есть."""
    rivers = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower().endswith(('.csv', '.json')):
                try:
                    with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as fh:
                        if f.endswith('.json'):
                            data = json.load(fh)
                            if isinstance(data, list):
                                rivers.extend(data)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    return rivers


@app.route('/api/water-stats')
def api_water_stats():
    """API для получения статистики паводков"""
    real_data = _load_water_data()
    fallback_rivers = [
        {'name': 'Лена', 'level': 420, 'norm': 400},
        {'name': 'Алдан', 'level': 310, 'norm': 300},
        {'name': 'Вилюй', 'level': 280, 'norm': 290},
        {'name': 'Колыма', 'level': 190, 'norm': 200},
        {'name': 'Яна', 'level': 150, 'norm': 160}
    ]
    rivers = real_data if real_data else fallback_rivers
    return jsonify({
        'current_level': 'Норма',
        'rivers': rivers,
        'is_fallback': not real_data
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'dataset-app'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8502, debug=False)
