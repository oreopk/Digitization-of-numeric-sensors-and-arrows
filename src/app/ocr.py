import easyocr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

reader = easyocr.Reader(['en'], gpu=True)


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