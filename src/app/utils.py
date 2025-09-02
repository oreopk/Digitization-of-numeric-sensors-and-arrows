import ctypes
import ctypes.wintypes
import cv2

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

def get_available_cameras_mini(max_index=5):
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cams.append(i)
        cap.release()
    return cams

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

