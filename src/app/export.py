import pandas as pd
from datetime import datetime

def export_to_excel(tracked_areas, history_lock):
    try:
        data = []
        with history_lock:
            for i, area in enumerate(tracked_areas):
                if area.history and area.history_timestamps:
                    for timestamp, value in zip(area.history_timestamps, area.history):
                        data.append({
                            "Area": i + 1,
                            "Time": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                            "Value": value
                        })
        
        if not data:
            print("Нет данных для экспорта")
            return
        
        df = pd.DataFrame(data)
        filename = f"ocr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Данные экспортированы в {filename}")
    except Exception as e:
        print(f"Ошибка при экспорте в Excel: {e}")