import cv2
import numpy as np
import easyocr
import re
import time
import threading
import pyttsx3
import subprocess
from collections import defaultdict
import language_conversion  

class MultiLangOCRSystem:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.")
            exit()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        self.tts_lock = threading.Lock()

        # Initialize OCR reader with multiple languages
        self.reader = easyocr.Reader(lang_list=["te", "en"], gpu=False)
        
        # Define script Unicode ranges
        self.script_ranges = {
            'ta': (0x0B80, 0x0BFF),
            'hi': (0x0900, 0x097F),
            'te': (0x0C00, 0x0C7F),
            'kn': (0x0C80, 0x0CFF),
            'ml': (0x0D00, 0x0D7F),
            'en': (0x0000, 0x007F)
        }

        self.last_spoken = ""
        self.confidence_threshold = 0.6   # Lowered for better detection
        self.process_interval = 0.5
        self.last_process_time = 0

        print("Multi-language OCR System initialized!")

    def optimize_image(self, image):
        """Enhance image quality for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced CLAHE intensity
        enhanced = clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (3, 3), 0)  # Use Gaussian blur instead of median

    def detect_script(self, text):
        """Detect dominant script using Unicode ranges"""
        script_counts = defaultdict(int)
        for char in text:
            code = ord(char)
            for lang, (start, end) in self.script_ranges.items():
                if start <= code <= end:
                    script_counts[lang] += 1
                    break
        
        return max(script_counts, key=script_counts.get, default='en')

    def process_text(self, text):
        """Clean OCR output"""
        return re.sub(r'[^\w\s\-.,!?]', '', text).strip()

    def speak_text(self, text):
        """Convert text to speech"""
        with self.tts_lock:  # Ensure only one speech thread at a time
            lang = self.detect_script(text)
            spoken_text = text
            print("Detected Text:", spoken_text)  # Debugging Output

            if language_conversion and lang != 'en':
                spoken_text = language_conversion.language_conversion(text, lang)
                print(spoken_text)
            self.engine.say(spoken_text)
            self.engine.runAndWait()

    def process_frame(self, frame):
        """Main OCR processing function"""
        current_time = time.time()
        if (current_time - self.last_process_time) < self.process_interval:
            return

        self.last_process_time = current_time
        processed_img = self.optimize_image(frame)

        # Perform OCR
        results = self.reader.readtext(
            processed_img,
            text_threshold=0.6,
            width_ths=0.7,
            add_margin=0.1
        )

        # Debugging: Print raw OCR results
        # print("OCR Results:", results)

        texts = [self.process_text(text) for _, text, conf in results if conf >= self.confidence_threshold]
        full_text = ' '.join(texts)

        # Debugging: Check what text is processed
        # print("Processed Text:", full_text)

        if full_text and full_text != self.last_spoken:
            self.last_spoken = full_text
            threading.Thread(target=self.speak_text, args=(full_text,)).start()

        return results

    def run(self):
        """Main capture loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Process frame
            results = self.process_frame(frame)

            if results:
                for (bbox, text, _) in results:
                    points = np.array(bbox).astype(np.int32)
                    cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                    cv2.putText(frame, text, tuple(points[0]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow('OCR System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ocr = MultiLangOCRSystem()
    ocr.run()
