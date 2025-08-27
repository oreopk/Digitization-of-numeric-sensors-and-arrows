

converted_history = []


plot_buffer = None

plot_ready = False
plot_width, plot_height = 0, 0
video_running = True

lower_h = 10
lower_s = 50
lower_v = 50
upper_h = 25
upper_s = 255
upper_v = 255

line_threshold = 300
min_line_length = 360
max_line_gap = 30


def run_dial_video_gui(cap):
    if isinstance(cap, int) or str(cap).isdigit():
        cap = cv2.VideoCapture(int(cap))
        if not cap.isOpened():
            print("Не удалось открыть камеру")
            return
    print(cap)
    import tkinter as tk
    from tkinter import ttk
    import cv2
    from PIL import Image, ImageTk
    import numpy as np
    from collections import deque
    from datetime import datetime
    import threading
    import time
    import pandas as pd
    from queue import Queue



    
    MAX_ANGLES = 1000
    angle_history = deque(maxlen=MAX_ANGLES)
    plot_lock = threading.Lock()
    frame_queue = Queue(maxsize=1)

    cap_lock = threading.Lock()

    def export_to_excel():
        if not converted_history:
            print("Нет данных для экспорта.")
            return
        df = pd.DataFrame(converted_history)
        filename = f"converted_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"Экспортировано: {filename}")

    def clear_plot_and_history():
        nonlocal angle_history 
        global converted_history, plot_buffer, plot_ready
        angle_history.clear()
        converted_history.clear()
        plot_buffer = None
        plot_ready = False
        print("График и история очищены.")


    def calculate_angle(line, center, long_part):
        x1, y1, x2, y2 = line
        if long_part == (x2, y2):
            angle_radians = np.arctan2(y1 - center[1], x1 - center[0])
        else:
            angle_radians = np.arctan2(y2 - center[1], x2 - center[0])
        return np.degrees(angle_radians)

    def create_opencv_plot(angles, width=400, height=300):
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        plot_img.fill(255)

        if not angles:
            return plot_img

        margin_left = 50
        margin_right = 20
        margin_top = 30
        margin_bottom = 50

        plot_area_width = width - margin_left - margin_right
        plot_area_height = height - margin_top - margin_bottom
        origin_y = margin_top + plot_area_height // 2

        cv2.line(plot_img, (margin_left, margin_top), 
                (margin_left, height - margin_bottom), (0, 0, 0), 2)
        cv2.line(plot_img, (margin_left, origin_y), 
                (width - margin_right, origin_y), (100, 100, 100), 1)

        max_angle = 90
        y_values = []
        for angle_data in angles:
            angle = angle_data['angle']
            clamped_angle = max(-max_angle, min(max_angle, angle))
            y = origin_y - int((clamped_angle / max_angle) * (plot_area_height / 2))
            y_values.append(y)

        for i in range(1, len(y_values)):
            x1 = margin_left + (i-1) * plot_area_width // len(y_values)
            x2 = margin_left + i * plot_area_width // len(y_values)
            cv2.line(plot_img, (x1, y_values[i-1]), (x2, y_values[i]), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4

        if len(angles) > 0:
            cv2.putText(plot_img, angles[0]['timestamp'], 
                    (margin_left, height - 10), font, font_scale, (0, 0, 0), 1)
            
            cv2.putText(plot_img, angles[-1]['timestamp'], 
                    (width - margin_right - 100, height - 10), font, font_scale, (0, 0, 0), 1)
            
            if len(angles) > 2:
                middle_idx = len(angles) // 2
                cv2.putText(plot_img, angles[middle_idx]['timestamp'], 
                        (margin_left + plot_area_width // 2 - 50, height - 10), 
                        font, font_scale, (0, 0, 0), 1)

        try:
            user_min = float(min_val_entry.get())
            user_max = float(max_val_entry.get())
        except:
            user_min, user_max = 0, 1000

        for frac, label_val in zip([0, 0.25, 0.5, 0.75, 1.0],
                                [user_min, user_min + 0.25*(user_max-user_min), 
                                user_min + 0.5*(user_max-user_min),
                                user_min + 0.75*(user_max-user_min), user_max]):
            y_pos = margin_top + int((1 - frac) * plot_area_height)
            cv2.putText(plot_img, f"{label_val:.0f}", (10, y_pos + 5), font, 0.5, (0, 0, 0), 1)

        return plot_img

    def update_plot():
        global plot_buffer, plot_ready, plot_width, plot_height
        while True:
            if len(angle_history) > 0 and plot_width > 0 and plot_height > 0:
                new_plot = create_opencv_plot(list(angle_history), plot_width, plot_height)
                with plot_lock:
                    plot_buffer = new_plot.copy()
                    plot_ready = True
            time.sleep(0.1)

    def line_intersects_point(x1, y1, x2, y2, px, py, threshold=50):
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return False
        proj = np.dot(point_vec, line_vec) / line_len
        proj = max(0, min(line_len, proj))
        closest = np.array([x1, y1]) + proj * line_vec / line_len
        distance = np.linalg.norm(closest - np.array([px, py]))
        return distance <= threshold

    def update_hsv():
        global lower_h, lower_s, lower_v, upper_h, upper_s, upper_v
        lower_h = lh.get(); lower_s = ls.get(); lower_v = lv.get()
        upper_h = uh.get(); upper_s = us.get(); upper_v = uv.get()

    def process_and_display():
        global plot_width, plot_height, prev_time
        
        frame_counter = 0
        prev_time = time.time()
        fps = 0
        fps_update_interval = 0.5
        last_angle_texts = ''
        last_fps_update = time.time()

        while True:
            angles_in_frame = []
            if not video_running:
                time.sleep(0.1)
                continue
            

            frame_counter += 1
            current_time = time.time()
            if current_time - last_fps_update >= fps_update_interval:
                fps = frame_counter / (current_time - last_fps_update)
                frame_counter = 0
                last_fps_update = current_time
                
            with cap_lock:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(cap, int) or str(cap).isdigit():
                        print("Ошибка захвата кадра с камеры")
                        time.sleep(0.1)
                        continue
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

            
            # fps = 1 / (current_time - prev_time)
            # prev_time = current_time
            if detection_enabled.get():
                smoothed_image = cv2.GaussianBlur(frame, (5, 5), 0)
                kernel = np.ones((5, 5), np.uint8)
                smoothed_image = cv2.morphologyEx(smoothed_image, cv2.MORPH_OPEN, kernel)
                smoothed_image = cv2.morphologyEx(smoothed_image, cv2.MORPH_CLOSE, kernel)

                hsv_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2HSV)
                lower_orange = np.array([lower_h, lower_s, lower_v])
                upper_orange = np.array([upper_h, upper_s, upper_v])
                orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
                orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
                orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
                inverted_orange_mask = cv2.bitwise_not(orange_mask)
                background_in_orange = cv2.bitwise_and(frame, frame, mask=inverted_orange_mask)
                contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                circular_mask = np.zeros_like(orange_mask)
                incenter = None
                circle_center = None
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(largest)
                    center = (int(x), int(y))
                    circle_center = center

                    big_radius = max(10, int(radius) - 10)
                    cv2.circle(circular_mask, center, big_radius, 255, -1)

                circular_mask = cv2.morphologyEx(circular_mask, cv2.MORPH_OPEN, kernel)
                circular_mask = cv2.morphologyEx(circular_mask, cv2.MORPH_CLOSE, kernel)
                final_result = cv2.bitwise_and(background_in_orange, background_in_orange, mask=circular_mask)
                final_result = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, kernel)
                final_result = cv2.morphologyEx(final_result, cv2.MORPH_CLOSE, kernel)

                plot_height, plot_width = final_result.shape[:2]

                final_contours, _ = cv2.findContours(cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                angles_in_frame = []

                if final_contours:
                    largest_final_contour = max(final_contours, key=cv2.contourArea)
                    final_with_contour = final_result.copy()
                    cv2.drawContours(frame, [largest_final_contour], -1, (0, 0, 255), 2)
                    (x,y), radius = cv2.minEnclosingCircle(largest_final_contour)
                    center = (int(x), int(y))
                    radius = max(10, int(radius))
                    cv2.circle(frame, center, radius, (0, 255, 255), 2)

                    mask = np.zeros(final_with_contour.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [largest_final_contour], -1, 255, -1)
                    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                    incenter = (max_loc[0], max_loc[1])
                    inradius = int(max_val)
                    cv2.circle(frame, incenter, inradius, (0, 255, 0), 2)
                    cv2.circle(frame, incenter, 3, (0, 255, 0), -1)
                    gray_contour = cv2.cvtColor(final_with_contour, cv2.COLOR_BGR2GRAY)
                    #lines = cv2.HoughLinesP(gray_contour, rho=1, theta=np.pi / 180, threshold=300, minLineLength=360, maxLineGap=30)
                    try:
                        th = int(threshold_scale.get())
                    except Exception:
                        th = 300
                    try:
                        ml = int(minlen_scale.get())
                    except Exception:
                        ml = 360
                    try:
                        mg = int(maxgap_scale.get())
                    except Exception:
                        mg = 30

                    lines = cv2.HoughLinesP(
                        gray_contour,
                        rho=1,
                        theta=np.pi / 180,
                        threshold=th,
                        minLineLength=ml,
                        maxLineGap=mg
                    )
                                
                    
                    if lines is not None:
                        filtered_lines = [line for line in lines if line_intersects_point(*line[0], *incenter)]
                        for line in filtered_lines:
                            x1, y1, x2, y2 = line[0]
                            cx, cy = incenter
                            length1 = np.hypot(x1 - cx, y1 - cy)
                            length2 = np.hypot(x2 - cx, y2 - cy)
                            long_part = (x1, y1) if length1 > length2 else (x2, y2)
                            short_part = (x2, y2) if long_part == (x1, y1) else (x1, y1)

                            angle = calculate_angle(line[0], (cx, cy), long_part)
                            if -80 < angle:
                                cv2.line(frame, (cx, cy), long_part, (0, 0, 255), 3)
                                cv2.line(frame, (cx, cy), short_part, (255, 0, 0), 3)
                                #if frame_counter % 1 == 0:
                                last_angle_texts =  f"{angle:.2f} deg"
                                cv2.putText(frame, last_angle_texts, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                #angles_in_frame.append(angle)
                                converted_value = get_mapped_value(angle)
                                converted_history.append({
                                    "Время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Значение": converted_value
                                })
                                angles_in_frame.append({
                                    'angle': angle,
                                    'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Формат: часы:минуты:секунды.миллисекунды
                                })
                            if show_angle_lines.get():
                                break
                    final_output = frame
                else:
                    final_output = final_result
            else:
                final_output = frame.copy() 

            if detection_enabled.get():
                if contours:
                    cv2.drawContours(final_output, [largest], -1, (0, 0, 255), 2)
                    cv2.circle(final_output, circle_center, big_radius, (0, 255, 255), 2)

                if circle_center:
                    cv2.circle(final_output, circle_center, 6, (255, 0, 255), -1)
                    cv2.circle(final_output, circle_center, 12, (255, 0, 255), 2)
                
                if incenter:
                    cv2.circle(final_output, incenter, 6, (255, 0, 255), -1)
                    cv2.circle(final_output, incenter, 12, (255, 0, 255), 2)

            plot_height, plot_width = final_output.shape[:2]
            if angles_in_frame:
                angle_history.extend(angles_in_frame)
                last_angle_data = angles_in_frame[-1]
                last_angle = last_angle_data['angle']
                mapped = get_mapped_value(last_angle)
                #last_angle = angles_in_frame[-1]
                #mapped = get_mapped_value(last_angle)
                text = f"Angle: {last_angle:.2f}"
                if mapped is not None:
                    text += f" Value {mapped:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = final_output.shape[1] - text_size[0] - 10
                text_y = 30
                cv2.putText(final_output, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

            if fps > 0:
                cv2.putText(final_output, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            #if detection_enabled.get():
            with plot_lock:
                current_plot = plot_buffer.copy() if plot_ready and plot_buffer is not None else np.ones_like(final_output) * 255
                current_plot = cv2.resize(current_plot, (final_output.shape[1], final_output.shape[0]))
                combined = np.hstack((final_output, current_plot))
                combined_rgb = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # else:
            #     combined = final_output.copy()
            #     combined_rgb = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
            MAX_WIDTH = 1200
            MAX_HEIGHT = 720
            ch, cw = combined_rgb.shape[:2]
            display_scale = min(MAX_WIDTH / cw, MAX_HEIGHT / ch, 1.0)
            if display_scale < 1.0:
                combined_rgb = cv2.resize(combined_rgb, (int(cw * display_scale), int(ch * display_scale)))
            
            frame_queue.put(ImageTk.PhotoImage(image=Image.fromarray(combined_rgb)))


            # root.after(16, process_and_display)


    def update_gui():
        if not frame_queue.empty():
            frame = frame_queue.get()
            video_label.imgtk = frame
            video_label.config(image=frame)
        if cap.isOpened():
            with cap_lock:
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                seek_percent = (current_frame / total_frames) * 100
                seek_var.set(seek_percent)
            else:
                seek_var.set(0)
        root.after(15, update_gui)


    def get_mapped_value(angle):
        try:
            min_val = float(min_val_entry.get())
            max_val = float(max_val_entry.get())
        except ValueError:
            return None
        input_min = -80
        input_max = 180
        if angle < input_min:
            return min_val
        elif angle > input_max:
            return max_val
        else:
            scale = (angle - input_min) / (input_max - input_min)
            return min_val + scale * (max_val - min_val)

    def toggle_video():
        global video_running
        video_running = not video_running
        toggle_btn.config(text="Продолжить" if not video_running else "Пауза")

    root = tk.Toplevel()
    root.title("HSV Analyzer")

    show_angle_lines = tk.BooleanVar(value=True)

    detection_enabled = tk.BooleanVar(value=False)

    def toggle_detection():
        detection_enabled.set(not detection_enabled.get())
        detect_btn.config(text="Включить анализ" if not detection_enabled.get() else "Отключить анализ")

    video_label = tk.Label(root)
    video_label.pack(side=tk.LEFT)


    controls = tk.Frame(root)
    controls.pack(side=tk.RIGHT, fill=tk.Y)

    hsv_controls = tk.Frame(root)
    hsv_controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10)


    tk.Button(controls, text="Очистить график", command=clear_plot_and_history).pack(pady=10)

    detect_btn = tk.Button(controls, text="Включить анализ", command=toggle_detection)
    detect_btn.pack(pady=10)

    toggle_btn = tk.Button(controls, text="Пауза", command=toggle_video)
    toggle_btn.pack(pady=10)
    
    tk.Label(controls, text="Мин. значение").pack()
    min_val_entry = tk.Entry(controls)
    min_val_entry.pack()
    min_val_entry.insert(0, "0")

    tk.Label(controls, text="Макс. значение").pack()
    max_val_entry = tk.Entry(controls)
    max_val_entry.pack()
    max_val_entry.insert(0, "1000")

    tk.Label(controls, text="Hough threshold").pack()
    threshold_scale = tk.Scale(controls, from_=1, to=1000, orient=tk.HORIZONTAL,
                            label="threshold", command=lambda val: None)
    threshold_scale.set(300)
    threshold_scale.pack()
    
    tk.Label(controls, text="Hough minLineLength").pack()
    minlen_scale = tk.Scale(controls, from_=1, to=1000, orient=tk.HORIZONTAL,
                            label="minLineLength", command=lambda val: None)
    minlen_scale.set(360)
    minlen_scale.pack()

    tk.Label(controls, text="Hough maxLineGap").pack()
    maxgap_scale = tk.Scale(controls, from_=0, to=500, orient=tk.HORIZONTAL,
                            label="maxLineGap", command=lambda val: None)
    maxgap_scale.set(30)
    maxgap_scale.pack()


    angle_check = tk.Checkbutton(controls, text="Только одна линия", 
                            variable=show_angle_lines, command=update_hsv)
    angle_check.pack()

    seek_var = tk.DoubleVar()
    seek_scale = tk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL, variable=seek_var, label="Перемотка (%)")
    seek_scale.pack()

    def on_seek(val):
        try:
            if not isinstance(cap, int) and not str(cap).isdigit():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                seek_percent = float(val)
                target_frame = int((seek_percent / 100) * total_frames)
                with cap_lock:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        except Exception as e:
            print("Ошибка при перемотке:", e)

    seek_scale.config(command=on_seek)




    tk.Button(controls, text="Сохранить в Excel", command=export_to_excel).pack(pady=10)



    lh = tk.Scale(hsv_controls, from_=0, to=180, orient=tk.HORIZONTAL, label="Lower H", command=lambda val: update_hsv())
    lh.set(lower_h); lh.pack()

    ls = tk.Scale(hsv_controls, from_=0, to=255, orient=tk.HORIZONTAL, label="Lower S", command=lambda val: update_hsv())
    ls.set(lower_s); ls.pack()

    lv = tk.Scale(hsv_controls, from_=0, to=255, orient=tk.HORIZONTAL, label="Lower V", command=lambda val: update_hsv())
    lv.set(lower_v); lv.pack()

    uh = tk.Scale(hsv_controls, from_=0, to=180, orient=tk.HORIZONTAL, label="Upper H", command=lambda val: update_hsv())
    uh.set(upper_h); uh.pack()

    us = tk.Scale(hsv_controls, from_=0, to=255, orient=tk.HORIZONTAL, label="Upper S", command=lambda val: update_hsv())
    us.set(upper_s); us.pack()

    uv = tk.Scale(hsv_controls, from_=0, to=255, orient=tk.HORIZONTAL, label="Upper V", command=lambda val: update_hsv())
    uv.set(upper_v); uv.pack()
        
    threading.Thread(target=update_plot, daemon=True).start()
    threading.Thread(target=process_and_display, daemon=True).start()

    update_gui()
    root.mainloop()
