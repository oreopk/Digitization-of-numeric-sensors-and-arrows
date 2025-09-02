import cv2

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