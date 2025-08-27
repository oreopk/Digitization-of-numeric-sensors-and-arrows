from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=30)

from .video_gui import run_dial_video_gui
from multiprocessing import cpu_count
from collections import deque
from PIL import Image, ImageTk
import cv2
import easyocr
import pytesseract
import numpy as np
import time
import tkinter as tk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from queue import Queue
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
from itertools import count
from tkinter import filedialog
from tkinter import messagebox
import ctypes
import ctypes.wintypes
import os
import psutil
import signal

# !!!!Указываем путь к Tesseract!!!!
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

reader = easyocr.Reader(['en'], gpu=True)

@dataclass
class TrackedArea:
    coords: list
    tracker: object = None
    active: bool = False
    last_text: str = ""
    history: deque = None
    history_timestamps: deque = None
    local_contrast: float = None
    rotation_angle: float = 0.0
    def __post_init__(self):
        self.history = deque(maxlen=1000)
        self.history_timestamps = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.last_update_time_find_number: int = 0
        self.rotation_angle = 0.0

class ResizeMode:
    CREATE = 1
    NONE = 0
    MOVE = 1
    TOP_LEFT = 2
    TOP_RIGHT = 3
    BOTTOM_LEFT = 4
    BOTTOM_RIGHT = 5
    TOP = 6
    BOTTOM = 7
    LEFT = 8
    RIGHT = 9


seek_slider = None

seek_var = None
seek_slider = None
time_label = None

total_frames = 0
current_frame_pos = 0

processed_rois_lock = threading.Lock()
current_resize_mode = ResizeMode.NONE
resize_area_index = -1
resize_start_coords = None
status_video = False
processing_active = False
result_counter = count(1)
MAX_AREAS = 6
processed_rois = [None] * MAX_AREAS
tracked_areas = []
current_area_index = 0
video_active = True
MAX_QUEUE_SIZE = 10000
current_frame = None
current_frame2 = None
fps_text = "FPS: calculating.."
contrast = 2.1
binary_value = 127
root_window = None
task_queue = Queue()
result_queue = Queue()
history_lock = threading.Lock()
recognition_labels = []
recognition_images = []
last_update_time_update_recognition_display = 0
last_update_time_update_results = 0
last_plot_update_time = 0
tracking_enabled = False
enable_binary = False
change_image_enabled = None
show_recognition_display = True
show_recognition_var = None


VIDEO_PATHS = []
SPECIAL_VIDEO_PATH = None


def toggle_recognition_display(show):
    global show_recognition_display
    show_recognition_display = show
    if show:
        update_recognition_display()



def center_window_winapi(window_name):
    try:
        user32 = ctypes.WinDLL('user32')
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            window_width = rect.right - rect.left
            window_height = rect.bottom - rect.top
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            user32.SetWindowPos(
                hwnd, 
                0,
                x, y, 
                0, 0,
                0x0001 | 0x0004
            )
    except:
        pass

def get_available_cameras(max_tests=5):
    available = []
    for i in range(max_tests):
        for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(i, api)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                    cap.release()
                    break
                cap.release()
            except:
                continue
    return available

def select_video_source():
    global VIDEO_PATHS, SPECIAL_VIDEO_PATH

    def get_available_cameras(max_index=5):
        cams = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cams.append(i)
            cap.release()
        return cams

    source_window = tk.Tk()
    source_window.title("Выбор источников видео")
    source_window.geometry("700x500")
    center_window_winapi("Выбор источников видео")

    notebook = ttk.Notebook(source_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    available_cameras = get_available_cameras()
    camera_vars = []
    file_vars = []
    special_camera_var = tk.StringVar(value="")
    special_video_path = tk.StringVar(value="")

    # === Вкладка Основные видео ===
    main_tab = ttk.Frame(notebook)
    notebook.add(main_tab, text="Числовые датчики")

    cameras_frame = ttk.LabelFrame(main_tab, text="Камеры")
    cameras_frame.pack(fill="x", padx=10, pady=5)

    for cam_idx in available_cameras:
        var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(cameras_frame, text=f"Камера {cam_idx}", variable=var)
        chk.pack(anchor="w", padx=20)
        camera_vars.append((cam_idx, var))

    files_frame = ttk.LabelFrame(main_tab, text="Видеофайлы")
    files_frame.pack(fill="x", padx=10, pady=5)

    def add_file():
        file_paths = filedialog.askopenfilenames(
            title="Выберите видеофайлы",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        for path in file_paths:
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(files_frame, text=path.split("/")[-1], variable=var)
            chk.pack(anchor="w", padx=20)
            file_vars.append((path, var))

    ttk.Button(files_frame, text="Добавить файлы...", command=add_file).pack(pady=5)

    # === Вкладка Специальное видео ===
    special_tab = ttk.Frame(notebook)
    notebook.add(special_tab, text="Циферблат")

    def select_special_video():
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if path:
            special_video_path.set(path)
            special_camera_var.set("")
            special_label.config(text=f"Выбрано видео: {path.split('/')[-1]}")

    ttk.Label(special_tab, text="Выберите один источник:").pack(pady=5)

    camera_frame = ttk.LabelFrame(special_tab, text="Камеры")
    camera_frame.pack(fill="x", padx=10, pady=5)

    for cam_idx in available_cameras:
        rb = ttk.Radiobutton(camera_frame, text=f"Камера {cam_idx}",
                             variable=special_camera_var, value=str(cam_idx),
                             command=lambda: special_video_path.set(""))
        rb.pack(anchor="w", padx=20)

    ttk.Button(special_tab, text="Выбрать видеофайл...", command=select_special_video).pack(pady=5)
    special_label = ttk.Label(special_tab, text="Ничего не выбрано", wraplength=400)
    special_label.pack(pady=5)

    # === Подтверждение ===
    confirm_frame = ttk.Frame(source_window)
    confirm_frame.pack(fill="x", padx=10, pady=10)

    def on_confirm():
        global VIDEO_PATHS, SPECIAL_VIDEO_PATH
        VIDEO_PATHS = []

        for cam_idx, var in camera_vars:
            if var.get():
                VIDEO_PATHS.append(cam_idx)
        for path, var in file_vars:
            if var.get():
                VIDEO_PATHS.append(path)

        if special_video_path.get():
            SPECIAL_VIDEO_PATH = special_video_path.get()
        elif special_camera_var.get() != "":
            try:
                SPECIAL_VIDEO_PATH = int(special_camera_var.get())
            except ValueError:
                SPECIAL_VIDEO_PATH = None
        else:
            SPECIAL_VIDEO_PATH = None

        print(">>> [DEBUG] SPECIAL_VIDEO_PATH =", SPECIAL_VIDEO_PATH)

        if not VIDEO_PATHS and SPECIAL_VIDEO_PATH is None:
            messagebox.showwarning("Ошибка", "Выберите хотя бы один источник.")
            return

        source_window.destroy()
        start_main_program()

    ttk.Button(confirm_frame, text="Подтвердить и запустить", command=on_confirm).pack(side="right")

    source_window.mainloop()


def open_video_source(source):
    if isinstance(source, int):
        for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(source, api)
            if cap.isOpened():
                return cap
            cap.release()
        return None
    else:
        return cv2.VideoCapture(source)

def start_main_program():
    global VIDEO_PATHS, SPECIAL_VIDEO_PATH
    

    if not VIDEO_PATHS and SPECIAL_VIDEO_PATH is None:
        print("Нет выбранных источников видео! Окно не будет запущено.")
        return
    root = None
    if VIDEO_PATHS:
        root = create_empty_window()
        update_recognition_display()
        update_results()
        update_plots()

        video_thread = threading.Thread(target=main, daemon=True)
        video_thread.start()

    if SPECIAL_VIDEO_PATH is not None:
        cap = open_video_source(SPECIAL_VIDEO_PATH)
        if cap and cap.isOpened():
            hidden_root = tk.Tk()
            hidden_root.withdraw()  # скрыть главное окно
            run_dial_video_gui(cap)

    
    if root:
        root.mainloop()

def show_special_video():
    global SPECIAL_VIDEO_PATH

    print(">>> show_special_video STARTED")
    print(">>> SPECIAL_VIDEO_PATH =", SPECIAL_VIDEO_PATH)

    cap = open_video_source(SPECIAL_VIDEO_PATH)

    if cap is None:
        print("[ОШИБКА] cap = None. Источник не открыт.")
        return

    print(">>> cap объект получен:", cap)
    print(">>> cap.isOpened() =", cap.isOpened())

    if isinstance(SPECIAL_VIDEO_PATH, int):
        for i in range(30):
            if cap.isOpened():
                break
            print(f"[INFO] Ожидание открытия камеры... попытка {i+1}")
            time.sleep(0.1)

    if not cap.isOpened():
        print(f"[ОШИБКА] Источник {SPECIAL_VIDEO_PATH} не открылся.")
        return

    cv2.namedWindow("Специальное видео", cv2.WINDOW_NORMAL)
    print(">>> Видеопоток запущен, читаем кадры...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ОШИБКА] Не удалось прочитать кадр. Завершение.")
            break

        cv2.imshow("Специальное видео", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(">>> Нажата 'q'. Выход из показа спецвидео.")
            break

    cap.release()
    cv2.destroyWindow("Специальное видео")
    print(">>> Поток спецвидео завершён")

def export_to_excel():
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


def init_tracker(frame, bbox):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"Неверный размер bbox: ширина={w}, высота={h}")
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, bbox)
    return tracker


def adjust_contrast(frame, contrast=2.1):
    frame = frame.astype('float32')
    frame = frame * contrast
    frame = np.clip(frame, 0, 255)
    return frame.astype('uint8')

def process_image(roi, area_id, timestamp):
    try:
        text1 = pytesseract.image_to_string(roi,
                    config=r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789').strip()
        if len(text1) == 4:
            results = reader.readtext(roi, detail=0, paragraph = False, batch_size=4, #contrast_ths=0.1,adjust_contrast=0.5, decoder='beamsearch',
                allowlist='0123456789') #adjust_contrast=0.5) , decoder='greedy'  batch_size=4
            text2 = ''.join(results).strip()
            if text1 == text2:
                return area_id, str(text1), timestamp

        results = reader.readtext(roi,detail=0, paragraph = False, batch_size=4, #contrast_ths=0.1,adjust_contrast=0.5, decoder='beamsearch',
                allowlist='0123456789')
        text2 = ''.join(results).strip()
        if len(text2) == 4:
            text1 = pytesseract.image_to_string(roi,
                    config=r'--oem 3 --psm 8 tessedit_char_whitelist=0123456789').strip()
            if text1 == text2:
                return area_id, str(text2), timestamp
        return area_id, "", timestamp
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return area_id, "", timestamp

def worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        roi, area_id, timestamp = task
        result = process_image(roi, area_id, timestamp)
        add_count = next(result_counter);
        result_queue.put((add_count, result))

        if task_info_label:
            task_info_label.config(text=f"Задачи: {task_queue.qsize()} | Обработано: {add_count}")

def start_workers():
    num_workers = max(1, cpu_count() - 1)
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    return threads

def update_results():
    global last_update_time_update_results

    processed_count = 0

    current_time = time.time()
    if current_time - last_update_time_update_results < 0.1:
        root_window.after(10, update_results)  # Исправлено: используем root_window вместо root
        return
    
    last_update_time_update_results = current_time

    while not result_queue.empty():
        item_number, (area_id, text, timestamp) = result_queue.get()
        processed_count += 1
        
        if status_label:
            status_label.config(text=f"Обрабатываю результат №{item_number}")
        
        if area_id < len(tracked_areas) and text:
            area = tracked_areas[area_id]
            area.last_text = text
            try:
                num = int(text)
                if num <= max_value_var.get():
                    with area.lock:
                        area.history.append(num)
                        area.history_timestamps.append(timestamp)
                else:
                    print(f"[INFO] Значение {num} выше порога, пропущено")
            except ValueError:
                pass
    
    if processed_count > 0 and status_label:
        status_label.config(text=f"Обработано {processed_count} результатов")

    root_window.after(10, update_results)  # Используем root_window вместо root

def clear_history_and_plots():
    global tracked_areas
    with history_lock:
        for area in tracked_areas:
            with area.lock:
                area.history.clear()
                area.history_timestamps.clear()
    for ax in axs:
        ax.clear()
    canvas.draw()



def update_plots():
    global last_plot_update_time

    current_time = time.time()
    if current_time - last_plot_update_time < 1:
        root_window.after(10, update_plots)
        return

    last_plot_update_time = current_time

    fig.clear()
    axs.clear()

    for i, area in enumerate(tracked_areas):
        with area.lock:
            if len(area.history) > 0:
                ax = fig.add_subplot(len(tracked_areas), 1, i + 1)
                axs.append(ax)

                timestamps, values = zip(*sorted(zip(area.history_timestamps, area.history)))
                ax.plot(timestamps, values, 'b-', marker='o')
                ax.set_title(f"Area {i+1} History")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.grid(True)

                if len(timestamps) > 1:
                    ax.xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: time.strftime('%H:%M:%S', time.localtime(x))))

    if axs:
        fig.tight_layout()
        canvas.draw()

    root_window.after(10, update_plots)


def change_image(image, use_easyocr=False, roi=True, contrast_value=None):
    global binary_value, contrast
    #inverted_roi = cv2.bitwise_not(image)
    fixed = image

    if contrast_value is None:
        contrast_value = contrast


    adjusted_roi = adjust_contrast(fixed, contrast_value)
    gray = cv2.cvtColor(adjusted_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)
    if enable_binary:
        if binary_value == 0:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, binary_value, 255, cv2.THRESH_BINARY)
        return binary
    else:
        return blurred


def process_roi_async(frame, area_id):
    global change_image_enabled
    x1, y1, x2, y2 = tracked_areas[area_id].coords
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2].copy()
        angle = tracked_areas[area_id].rotation_angle
        if angle != 0:
            (h, w) = roi.shape[:2]
            center = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            roi = cv2.warpAffine(roi, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if roi.size == 0:
            return
        
        scaled_roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if change_image_enabled.get():
            #processed_roi = change_image(scaled_roi, False, False)
            contrast_value = tracked_areas[area_id].local_contrast
            processed_roi = change_image(scaled_roi, False, False, contrast_value)
        else:
            processed_roi = scaled_roi.copy()
        timestamp = time.time()

        # Обнаружение текста
        results = reader.readtext(processed_roi, detail=1, paragraph=False, allowlist='0123456789', adjust_contrast=0.7)

        selected_crop = None
        for bbox, text, conf in results:
            if len(text.strip()) == 4:
                pts = np.array(bbox, dtype=np.int32)
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                cv2.polylines(processed_roi, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # Вырезаем найденную строку
                selected_crop = processed_roi[y_min:y_max, x_min:x_max]
                break

        # if selected_crop is not None:
        #     data = pytesseract.image_to_data(processed_roi, config='--psm 6 -c tessedit_char_whitelist=0123456789', output_type=pytesseract.Output.DICT)
        #     for i in range(len(data['text'])):
        #         text = data['text'][i].strip()
        #         if len(text) == 4:
        #             x = data['left'][i]
        #             y = data['top'][i]
        #             w = data['width'][i]
        #             h = data['height'][i]

        #             # Вырезаем участок
        #             tess_crop = processed_roi[y:y+h, x:x+w]
        #             selected_crop = tess_crop.copy()
        #             # Визуально выделим прямоугольник
        #             cv2.rectangle(processed_roi, (x, y), (x + w, y + h), (255, 0, 0), 4)

        #             break  # если надо только первое совпадение

        with processed_rois_lock:
                processed_rois[area_id] = processed_roi.copy()  # fallback

        if processing_active:
            if selected_crop is not None and selected_crop.size > 0:
                task_queue.put((selected_crop, area_id, timestamp))


def mouse_callback(event, x, y, flags, param):
    global current_area_index, tracked_areas, current_resize_mode, resize_area_index, resize_start_coords

    if current_resize_mode != ResizeMode.NONE and flags & cv2.EVENT_FLAG_LBUTTON:
        if 0 <= resize_area_index < len(tracked_areas):
            area = tracked_areas[resize_area_index]
            x1, y1, x2, y2 = area.coords

            if current_resize_mode == ResizeMode.MOVE:
                dx = x - resize_start_coords[0]
                dy = y - resize_start_coords[1]
                new_coords = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            else:
                new_coords = [x1, y1, x2, y2]
                if current_resize_mode in [ResizeMode.TOP_LEFT, ResizeMode.LEFT, ResizeMode.BOTTOM_LEFT]:
                    new_coords[0] = x
                if current_resize_mode in [ResizeMode.TOP_RIGHT, ResizeMode.RIGHT, ResizeMode.BOTTOM_RIGHT]:
                    new_coords[2] = x
                if current_resize_mode in [ResizeMode.TOP_LEFT, ResizeMode.TOP, ResizeMode.TOP_RIGHT]:
                    new_coords[1] = y
                if current_resize_mode in [ResizeMode.BOTTOM_LEFT, ResizeMode.BOTTOM, ResizeMode.BOTTOM_RIGHT]:
                    new_coords[3] = y

            if abs(new_coords[2] - new_coords[0]) > 10 and abs(new_coords[3] - new_coords[1]) > 10:
                area.coords = new_coords
                resize_start_coords = (x, y)

                x1, y1, x2, y2 = area.coords
                w, h = x2 - x1, y2 - y1
                try:
                    area.tracker = init_tracker(current_frame, (x1, y1, w, h))
                except ValueError as e:
                    print(f"Ошибка: {e}. Пропускаем некорректный bbox.")

        return

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, area in enumerate(tracked_areas):
            x1, y1, x2, y2 = area.coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                resize_area_index = i
                resize_start_coords = (x, y)
                current_resize_mode = ResizeMode.MOVE
                break

    elif event == cv2.EVENT_LBUTTONUP:
        current_resize_mode = ResizeMode.NONE
        resize_area_index = -1
        resize_start_coords = None

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, area in enumerate(tracked_areas):
            if area.coords:
                x1, y1, x2, y2 = area.coords
                margin = 10
 
                if abs(x - x1) < margin and abs(y - y1) < margin:
                    current_resize_mode = ResizeMode.TOP_LEFT
                elif abs(x - x2) < margin and abs(y - y1) < margin:
                    current_resize_mode = ResizeMode.TOP_RIGHT
                elif abs(x - x1) < margin and abs(y - y2) < margin:
                    current_resize_mode = ResizeMode.BOTTOM_LEFT
                elif abs(x - x2) < margin and abs(y - y2) < margin:
                    current_resize_mode = ResizeMode.BOTTOM_RIGHT
                elif abs(y - y1) < margin and x1 <= x <= x2:
                    current_resize_mode = ResizeMode.TOP
                elif abs(y - y2) < margin and x1 <= x <= x2:
                    current_resize_mode = ResizeMode.BOTTOM
                elif abs(x - x1) < margin and y1 <= y <= y2:
                    current_resize_mode = ResizeMode.LEFT
                elif abs(x - x2) < margin and y1 <= y <= y2:
                    current_resize_mode = ResizeMode.RIGHT
                elif x1 <= x <= x2 and y1 <= y <= y2:
                    current_resize_mode = ResizeMode.MOVE
                else:
                    continue
                
                resize_area_index = i
                resize_start_coords = (x, y)
                current_area_index = i
                return

        if len(tracked_areas) < MAX_AREAS:
            tracked_areas.append(TrackedArea(coords=[x, y, x, y]))
            current_area_index = len(tracked_areas) - 1
        else:
            current_area_index = (current_area_index + 1) % MAX_AREAS
            tracked_areas[current_area_index].coords = [x, y, x, y]
    
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if current_resize_mode == ResizeMode.NONE and tracked_areas and current_area_index < len(tracked_areas):
            tracked_areas[current_area_index].coords[2] = x
            tracked_areas[current_area_index].coords[3] = y
    
    elif event == cv2.EVENT_LBUTTONUP:
        if current_resize_mode != ResizeMode.NONE:
            current_resize_mode = ResizeMode.NONE
            resize_area_index = -1
        elif tracked_areas and current_area_index < len(tracked_areas):
            area = tracked_areas[current_area_index]
            x1, y1, x2, y2 = area.coords
            w, h = abs(x2 - x1), abs(y2 - y1)

            if w > 10 and h > 10:
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                area.coords = [x1, y1, x2, y2]
                area.tracker = init_tracker(current_frame, (x1, y1, w, h))
                area.active = True
                update_plots()

def update_tracking(frame):
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


def on_seek(val):
    global current_frame_pos, caps, status_video
    if not caps:
        return

    was_running = status_video
    status_video = False  # временно приостанавливаем воспроизведение

    seek_pos = float(val)
    for cap in caps:
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_pos = int((seek_pos / 100) * total)

            if frame_pos <= 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
                cap.read()  # стабилизация после seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    current_frame_pos = frame_pos
    update_time_label()

    if was_running:
        status_video = True

def update_time_label():
    global caps, time_label, seek_var
    if not caps or not caps[0].isOpened():
        return
    
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    current_time = current_frame_pos / fps
    total_time = total_frames / fps
    
    current_str = time.strftime('%M:%S', time.gmtime(current_time))
    total_str = time.strftime('%M:%S', time.gmtime(total_time))
    
    time_label.config(text=f"{current_str} / {total_str}")
    if seek_var is not None:
        seek_var.set((current_frame_pos / total_frames) * 100 if total_frames > 0 else 0)



def update_recognition_display():
    global last_update_time_update_recognition_display, show_recognition_display
    
    if not show_recognition_display:
        root_window.after(10, update_recognition_display)
        return
    
    current_time = time.time()
    if current_time - last_update_time_update_recognition_display < 0.1:
        root_window.after(10, update_recognition_display)
        return
    
    last_update_time_update_recognition_display = current_time

    for i, area in enumerate(tracked_areas):
        if area.coords:
            with processed_rois_lock:
                roi = processed_rois[i]
                
            if roi is not None and roi.size > 0:
                try:
                    # Конвертируем изображение в формат для Tkinter
                    if len(roi.shape) == 2:
                        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                    else:
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    
                    # Масштабируем изображение для лучшего отображения
                    h, w = roi.shape[:2]
                    scale_factor = min(200/h, 200/w)
                    if scale_factor < 1:  # Уменьшаем только большие изображения
                        roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor)
                    
                    img_pil = Image.fromarray(roi)
                    img_tk = ImageTk.PhotoImage(image=img_pil)

                    recognition_labels[i].config(text=f"Area {i+1}: {area.last_text}")
                    recognition_images[i].config(image=img_tk)
                    recognition_images[i].image = img_tk  # Сохраняем ссылку!
                    
                except Exception as e:
                    print(f"Ошибка обновления отображения для области {i+1}: {e}")

    root_window.after(10, update_recognition_display)

def start():
    global processing_active
    processing_active = True

def stop():
    global processing_active
    processing_active = False

def start_video():
    global status_video
    status_video = True

def stop_video():
    global status_video
    status_video = False

def create_empty_window():
    global root_window, fig, axs, canvas, status_label, task_info_label, enable_binary_var
    global seek_slider, time_label, seek_var, change_image_enabled, max_value_var

    root = tk.Tk()
    root_window = root
    root.title("Control Panel")
    root.geometry("1600x900")
    

    seek_var = tk.DoubleVar()

    def on_closing():
        global caps
        cv2.destroyAllWindows()
        if caps:
            for cap in caps:
                cap.release()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 9), padding=10)
    

    control_frame = ttk.Frame(root)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=3, pady=3)

    def update_contrast(val):
        global contrast
        contrast = float(val)
        #contrast_label.config(text=str(contrast))

    def update_binary(val):
        global binary_value
        binary_value = int(float(val))
        if enable_binary_var.get():
            binary_slider.config(state=tk.NORMAL)
        else:
            binary_slider.config(state=tk.DISABLED)
    
    def toggle_tracking():
        global tracking_enabled
        tracking_enabled = tracking_var.get()

    def toggle_binary():
        global enable_binary
        enable_binary = enable_binary_var.get()
        if enable_binary:
            binary_slider.config(state=tk.NORMAL)
        else:
            binary_slider.config(state=tk.DISABLED)

    def close_all():
        os._exit(0)

    change_image_enabled = tk.BooleanVar(value=True)

    change_image_checkbox = ttk.Checkbutton(
        control_frame,
        text="Предобработка",
        variable=change_image_enabled,
        onvalue=True,
        offvalue=False
    )
    change_image_checkbox.pack(side=tk.LEFT, padx=10)
    tracking_var = tk.BooleanVar(value=False)
    tracking_checkbox = ttk.Checkbutton(control_frame, text="Включить трекинг", variable=tracking_var, command=toggle_tracking)
    tracking_checkbox.pack(side=tk.LEFT, padx=10)

    contrast_frame = ttk.Frame(control_frame)
    contrast_frame.pack(side=tk.LEFT, padx=10)
    

    enable_binary_var = tk.BooleanVar(value=False)
    binary_check = ttk.Checkbutton(
        control_frame, 
        text="Включить бинаризацию", 
        variable=enable_binary_var,
        command=toggle_binary
    )
    binary_check.pack(side=tk.LEFT, padx=10)



    binary_frame = ttk.Frame(control_frame)
    binary_frame.pack(side=tk.LEFT, padx=10)


    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(side=tk.LEFT, padx=10)
    
    start_btn = ttk.Button(btn_frame, text="Start Processing", command=start)
    start_btn.pack(side=tk.LEFT, padx=5)
    
    stop_btn = ttk.Button(btn_frame, text="Stop Processing", command=stop)
    stop_btn.pack(side=tk.LEFT, padx=5)

    btn_frame2 = ttk.Frame(control_frame)
    btn_frame2.pack(side=tk.LEFT, padx=10)

    max_value_frame = ttk.Frame(control_frame)
    max_value_frame.pack(side=tk.LEFT, padx=10)

    ttk.Label(max_value_frame, text="Макс. значение").pack(side=tk.LEFT)

    max_value_var = tk.IntVar(value=5000)
    max_value_entry = ttk.Entry(max_value_frame, textvariable=max_value_var, width=6)
    max_value_entry.pack(side=tk.LEFT)

    start_btn_video = ttk.Button(btn_frame2, text="Start Video", command=start_video)
    start_btn_video.pack(side=tk.LEFT, padx=5)
    
    stop_btn_video = ttk.Button(btn_frame2, text="Stop Video", command=stop_video)
    stop_btn_video.pack(side=tk.LEFT, padx=5)

    contrast_label = ttk.Label(contrast_frame, text="Contrast")
    contrast_label.pack(side=tk.LEFT)
    
    contrast_slider = ttk.Scale(contrast_frame, from_=0.1, to=3.0, value=2.2, command=update_contrast)
    contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    # contrast_label = ttk.Label(contrast_frame, text="1")
    # contrast_label.pack(side=tk.LEFT)

    status_frame = ttk.Frame(root)
    status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    status_label = ttk.Label(status_frame, text="Статус: Ожидание", font=('Helvetica', 10))
    status_label.pack(side=tk.LEFT, padx=5)
    
    task_info_label = ttk.Label(status_frame, text="Задачи: 0 | Обработано: 0", font=('Helvetica', 10))
    task_info_label.pack(side=tk.LEFT, padx=5)

    export_button = ttk.Button(status_frame, text="Export to Excel", command=export_to_excel)
    export_button.pack(side=tk.LEFT, padx=10)

    clear_button = ttk.Button(status_frame, text="Очистить графики", command=clear_history_and_plots)
    clear_button.pack(side=tk.LEFT, padx=10)

    show_recognition_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        status_frame,
        text="Показывать распознавание",
        variable=show_recognition_var,
        command=lambda: toggle_recognition_display(show_recognition_var.get())
    ).pack(side=tk.LEFT, padx=10)

    btn_exit = ttk.Button(status_frame, text="Закрыть программу", command=close_all)
    btn_exit.pack(side=tk.LEFT, padx=10)

    seek_frame = ttk.Frame(status_frame)
    seek_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
    
    time_label = ttk.Label(seek_frame, text="00:00 / 00:00")
    time_label.pack(side=tk.RIGHT, padx=5)
    
    seek_slider = ttk.Scale(
        seek_frame, 
        from_=0, 
        to=100, 
        variable=seek_var,
        command=on_seek,
        orient=tk.HORIZONTAL
    )
    seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    binary_label = ttk.Label(binary_frame, text="Binary")
    binary_label.pack(side=tk.LEFT)

    binary_slider = ttk.Scale(binary_frame, from_=0,to=255, value=0,command=update_binary)
    binary_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    graph_frame = ttk.Frame(main_frame)
    graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # fig, axs = plt.subplots(MAX_AREAS, 1, figsize=(6, 3*MAX_AREAS))
    # if MAX_AREAS == 1:
    #     axs = [axs]
    fig = plt.figure(figsize=(6, 3))
    axs = []

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Recognition frame
    recognition_frame = ttk.Frame(main_frame, width=250)
    recognition_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

    global recognition_labels, recognition_images
    recognition_labels = []
    recognition_images = []

    for i in range(MAX_AREAS):
        area_frame = ttk.Frame(recognition_frame)
        area_frame.pack(fill=tk.X, pady=5)

        top_row = ttk.Frame(area_frame)
        top_row.pack(fill=tk.X)

        # Картинка слева
        img_label = ttk.Label(top_row)
        img_label.pack(side=tk.RIGHT, padx=5)
        recognition_images.append(img_label)

        slider_column = ttk.Frame(top_row)
        slider_column.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        label = ttk.Label(slider_column, text=f"Area {i+1}: Waiting...", font=('Helvetica', 10))
        label.pack(anchor=tk.W)
        recognition_labels.append(label)


        rotation_var = tk.DoubleVar(value=0.0)

        def make_rotation_callback(idx, var):
            def callback(val):
                tracked_areas[idx].rotation_angle = float(val)
            return callback

        
        contrast_var = tk.DoubleVar(value=2.1)

        def make_contrast_callback(idx, var):
            def callback(val):
                if idx < len(tracked_areas):
                    tracked_areas[idx].local_contrast = float(val)
            return callback

        ttk.Label(slider_column, text="Поворот (°)").pack(anchor=tk.W)
        rotation_slider = ttk.Scale(slider_column, from_=-180, to=180, variable=rotation_var,
                                    command=make_rotation_callback(i, rotation_var), orient=tk.HORIZONTAL)
        rotation_slider.pack(fill=tk.X)

        ttk.Label(slider_column, text="Контраст").pack(anchor=tk.W)
        contrast_slider = ttk.Scale(slider_column, from_=0.1, to=3.0, variable=contrast_var,
                            command=lambda val, idx=i: make_contrast_callback(idx, contrast_var)(round(float(val), 1)),
                            orient=tk.HORIZONTAL)
        contrast_slider.pack(fill=tk.X)

    return root

def main():
    global current_frame, video_active, fps_text, VIDEO_PATHS, caps, current_frame_pos

    if not VIDEO_PATHS:
        print("Нет выбранных источников видео!")
        return

    caps = []
    valid_sources = []
    for source in VIDEO_PATHS:
        cap = open_video_source(source)
        if cap and cap.isOpened():
            caps.append(cap)
            valid_sources.append(source)
            print(f"Успешно открыт источник: {source}")
        else:
            print(f"Не удалось открыть источник: {source}")

    if not caps:
        print("Ни один источник видео не открылся!")
        return

    VIDEO_PATHS = valid_sources

    workers = start_workers()
    #initial_width, initial_height = 1600, 520
    
    cv2.namedWindow('Multi Video Stream', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Multi Video Stream', mouse_callback)
    #cv2.resizeWindow('Multi Video Stream', initial_width, initial_height)
    center_window_winapi('Multi Video Stream')

    try:
        widths = []
        heights = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                widths.append(w)
                heights.append(h)
        total_width = sum(widths)
        total_height = max(heights) if heights else 480
        aspect_ratio = total_width / total_height if total_height else 1.0
        target_width = 1920
        target_height = int(target_width / aspect_ratio)
        cv2.resizeWindow('Multi Video Stream', target_width, target_height)
    except Exception as e:
        print("Ошибка при установке размера окна:", e)

    frame_count = 0
    last_fps_update = time.time()
    fps_text = "FPS: 0.0"
    DEFAULT_HEIGHT, DEFAULT_WIDTH = 480, 640
    separator_width = 5
    frames = []
    for i, cap in enumerate(caps):
        try:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
            frames.append(frame)
        except Exception as e:
            print(f"Ошибка чтения кадра с источника {VIDEO_PATHS[i]}: {e}")
            frame = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
            frames.append(frame)
    
    if caps and caps[0].isOpened():
            total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
            seek_slider.config(to=100)
            update_time_label()


    while True:
        start_time = time.time()
        
        if status_video:
            frames = []
            for i, cap in enumerate(caps):
                try:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            frame = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
                    frames.append(frame)
                    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                except Exception as e:
                    print(f"Ошибка чтения кадра с источника {VIDEO_PATHS[i]}: {e}")
                    frame = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
                    frames.append(frame)
            
            if root_window:
                root_window.after(0, update_time_label)

        # Если нет ни одного кадра, пропускаем итерацию
        if not frames:
            continue

        # Создаем комбинированный кадр
        if len(frames) == 1:
            combined_frame = frames[0]
        else:
            heights = [f.shape[0] for f in frames]
            widths = [f.shape[1] for f in frames]
            total_width = sum(widths) + separator_width * (len(frames) - 1)
            max_height = max(heights)

            combined_frame = np.zeros((max_height, total_width, 3), dtype=np.uint8)

            x_offset = 0
            for i, frame in enumerate(frames):
                h, w = frame.shape[:2]
                combined_frame[:h, x_offset:x_offset+w] = frame
                if i < len(frames) - 1:
                    # Добавляем разделитель
                    cv2.line(combined_frame, 
                            (x_offset + w, 0), 
                            (x_offset + w, max_height), 
                            (0, 255, 0), separator_width)
                    x_offset += w + separator_width

        display_frame = update_tracking(combined_frame.copy())

        with threading.Lock():
            for i, area in enumerate(tracked_areas):
                if area.coords:
                    cv2.rectangle(display_frame, 
                                (area.coords[0], area.coords[1]), 
                                (area.coords[2], area.coords[3]), 
                                (255, 255, 255) if i == current_area_index else (255, 255, 255), 2)
                    cv2.putText(display_frame, str(i+1), 
                              (area.coords[0] - 5, area.coords[1] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    angle = tracked_areas[i].rotation_angle
                    cv2.putText(display_frame, f"{i+1} ({int(angle)}°)",
                                (area.coords[0], area.coords[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    executor.submit(process_roi_async, display_frame.copy(), i)
                    #process_roi_async(display_frame, i)

        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_update >= 1:
            fps = frame_count / (current_time - last_fps_update)
            fps_text = f"FPS: {fps:.1f} | Areas: {len(tracked_areas)}/{MAX_AREAS}"
            frame_count = 0
            last_fps_update = current_time

        cv2.putText(display_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Multi Video Stream', display_frame)
        cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / 30) - elapsed_time)
        time.sleep(sleep_time)

        key = cv2.waitKey(1)
        if key == 27:
            break

    for cap in caps:
        cap.release()
    for _ in range(len(workers)):
        task_queue.put(None)
    cv2.destroyAllWindows()
    if root_window:
        root_window.quit()

if __name__ == "__main__":
    select_video_source()










