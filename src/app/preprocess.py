import cv2
import numpy as np

def adjust_contrast(frame, contrast=2.1):
    frame = frame.astype('float32')
    frame = frame * contrast
    frame = np.clip(frame, 0, 255)
    return frame.astype('uint8')


def change_image(image, enable_binary, binary_value, default_contrast, contrast_value=None):
    #inverted_roi = cv2.bitwise_not(image)
    fixed = image

    if contrast_value is None:
        contrast_value = default_contrast


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