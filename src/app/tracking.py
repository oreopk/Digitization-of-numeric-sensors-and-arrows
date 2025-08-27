import cv2

def init_tracker(frame, bbox):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"Неверный размер bbox: ширина={w}, высота={h}")
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, bbox)
    return tracker

def update_tracking(frame, tracked_areas, tracking_enabled):
    if not tracking_enabled:
        return frame
    for i, area in enumerate(tracked_areas):
        if area.tracker is None or not area.active:
            if area.coords:
                x1, y1, x2, y2 = area.coords
                w, h = x2 - x1, y2 - y1
                try:
                    area.tracker = init_tracker(frame, (x1, y1, w, h))
                    area.active = True
                except Exception as e:
                    print(f"Ошибка инициализации трекера для области {i}: {e}")
                    area.active = False
                    continue
        
        try:
            success, box = area.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                area.coords = [x, y, x + w, y + h]
            else:
                area.active = False
        except Exception as e:
            print(f"Ошибка обновления трекера для области {i}: {e}")
            area.active = False

    return frame