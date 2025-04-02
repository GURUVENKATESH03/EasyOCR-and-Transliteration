

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
        self.cap.set(cv2.CAP_PROP_FPS,20)
        self.speaking=False

        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        self.tts_lock = threading.Lock()

        # Initialize OCR reader with multiple languages
        self.reader = easyocr.Reader(lang_list=["kn", "en"], gpu=False)

        # Define script Unicode ranges
        self.script_ranges = {
            'te': (0x0C00, 0x0C7F),
            'kn': (0x0C80, 0x0CFF),
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
        self.speaking=True
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
            self.speaking=False

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
            add_margin=0.1,
            contrast_ths=0.5
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
        if not self.speaking:
            """Main capture loop"""
            while True:
                time.sleep(0.3)
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

# import cv2
# import numpy as np
# import easyocr
# import re
# import time
# import threading
# import pyttsx3
# import subprocess
# from collections import defaultdict
# import language_conversion
# import trasn

# class MultiLangOCRSystem:
#     def __init__(self):
#         # Initialize webcam
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             print("Error: Unable to access the webcam.")
#             exit()

#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self.cap.set(cv2.CAP_PROP_FPS,20)
#         self.speaking = False

#         # Initialize TTS engine
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', 130)
#         self.tts_lock = threading.Lock()

#         # Initialize OCR reader with multiple languages
#         self.reader = easyocr.Reader(lang_list=["te", "en"], gpu=False)

#         # Define script Unicode ranges
#         self.script_ranges = {
#             'ta': (0x0B80, 0x0BFF),
#             'hi': (0x0900, 0x097F),
#             'te': (0x0C00, 0x0C7F),
#             'kn': (0x0C80, 0x0CFF),
#             'ml': (0x0D00, 0x0D7F),
#             'en': (0x0000, 0x007F)
#         }

#         self.last_spoken = ""
#         self.confidence_threshold = 0.6   # Lowered for better detection
#         self.process_interval = 1
#         self.last_process_time = 0

#         print("Multi-language OCR System initialized!")

#     def optimize_image(self, image):
#         """Enhance image quality for better OCR results"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced CLAHE intensity
#         enhanced = clahe.apply(gray)
#         return cv2.GaussianBlur(enhanced, (3, 3), 0)  # Use Gaussian blur instead of median

#     def detect_script(self, text):
#         """Detect dominant script using Unicode ranges"""
#         script_counts = defaultdict(int)
#         for char in text:
#             code = ord(char)
#             for lang, (start, end) in self.script_ranges.items():
#                 if start <= code <= end:
#                     script_counts[lang] += 1
#                     break

#         return max(script_counts, key=script_counts.get, default='en')

#     def process_text(self, text):
#         """Clean OCR output"""
#         return re.sub(r'[^\w\s\-.,!?]', '', text).strip()

#     def speak_text(self, text):
#         """Convert text to speech"""
#         self.speaking = True
#         with self.tts_lock:  # Ensure only one speech thread at a time
#             lang = self.detect_script(text)
#             spoken_text = text
#             print("Detected Text:", spoken_text)  # Debugging Output

#             if language_conversion and lang != 'en':
#                 # spoken_text = language_conversion.language_conversion(text, lang)
#                 spoken_text = trasn.transliterate(text,lang)
#                 print(spoken_text)
#             self.engine.say(spoken_text)
#             self.engine.runAndWait()
#             self.speaking = False

#     def process_frame(self, frame):
#         """Main OCR processing function"""
#         current_time = time.time()
#         if (current_time - self.last_process_time) < self.process_interval:
#             return

#         self.last_process_time = current_time
#         processed_img = self.optimize_image(frame)

#         # Perform OCR
#         results = self.reader.readtext(
#             processed_img,
#             text_threshold=0.3,
#             width_ths=0.6,
#             add_margin=0.1
#         )

#         # Debugging: Print raw OCR results
#         # print("OCR Results:", results)

#         texts = [self.process_text(text) for _, text, conf in results if conf >= self.confidence_threshold]
#         full_text = ' '.join(texts)

#         # Debugging: Check what text is processed
#         # print("Processed Text:", full_text)

#         if full_text and full_text != self.last_spoken:
#             self.last_spoken = full_text
#             threading.Thread(target=self.speak_text, args=(full_text,)).start()

#         return results

#     def run(self):
#         """Main capture loop"""
#         if not self.speaking:
#             while True:
#                 time.sleep(2)
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     print("Error: Unable to capture frame.")
#                     break

#                 # Process frame
#                 results = self.process_frame(frame)

#                 if results:
#                     for (bbox, text, _) in results:
#                         points = np.array(bbox).astype(np.int32)
#                         cv2.polylines(frame, [points], True, (0, 255, 0), 2)
#                         cv2.putText(frame, text, tuple(points[0]),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#                 cv2.imshow('OCR System', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             self.cap.release()
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     ocr = MultiLangOCRSystem()
#     ocr.run()


# import cv2
# import numpy as np
# import easyocr
# import re
# import time
# import threading
# import pyttsx3
# import subprocess
# from collections import defaultdict
# import language_conversion

# class MultiLangOCRSystem:
#     def __init__(self):
#         # Initialize webcam
#         self.cap = cv2.VideoCapture(0)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         # Initialize TTS engine
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', 130)
#         self.tts_lock = threading.Lock()

#         # Initialize OCR reader with multiple languages
#         # self.languages = ['ta','en', 'te','kn']
#         self.reader = easyocr.Reader(lang_list=["te","en"], gpu=False)
#         # self.kannada_reader = easyocr.Reader(lang_list=["kn", "en"], gpu=False)

#         # Define script Unicode ranges
#         self.script_ranges = {
#             'ta': (0x0B80, 0x0BFF),
#             'hi': (0x0900, 0x097F),
#             'te': (0x0C00, 0x0C7F),
#             'kn': (0x0C80, 0x0CFF),
#             'ml': (0x0D00, 0x0D7F),
#             'en': (0x0000, 0x007F)
#         }

#         self.last_spoken = ""
#         self.confidence_threshold = 0.5  # Slightly increased for accuracy
#         self.process_interval = 0.5
#         self.last_process_time = 0

#         print("Multi-language OCR System initialized!")

#     def optimize_image(self, image):
#         """Enhance image quality for better OCR results"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#         return cv2.medianBlur(enhanced, 3)

#     def detect_script(self, text):
#         """Detect dominant script using Unicode ranges"""
#         script_counts = defaultdict(int)
#         for char in text:
#             code = ord(char)
#             for lang, (start, end) in self.script_ranges.items():
#                 if start <= code <= end:
#                     script_counts[lang] += 1
#                     break

#         return max(script_counts, key=script_counts.get, default='en')

#     def process_text(self, text):
#         """Clean OCR output"""
#         return re.sub(r'[^\w\s\-.,!?]', '', text).strip()

#     def speak_text(self, text):
#         """Convert text to speech"""
#         with self.tts_lock:  # Ensure only one speech thread at a time
#             lang = self.detect_script(text)
#             spoken_text = text
#             print(text)
#             if language_conversion and lang != 'en':
#                 spoken_text = language_conversion.language_conversion(text, lang)
#                 print("Detected Text "+spoken_text)

#             self.engine.say(spoken_text)
#             self.engine.runAndWait()  # Ensure speech finishes before proceeding
#             # subprocess.run(['espeak', '-s', '150', '-p', '50', spoken_text])

#     def process_frame(self, frame):
#         """Main OCR processing function"""
#         current_time = time.time()
#         if (current_time - self.last_process_time) < self.process_interval:
#             return

#         self.last_process_time = current_time
#         processed_img = self.optimize_image(frame)

#         # Perform OCR
#         results = self.reader.readtext(
#             processed_img,
#             text_threshold=0.6,
#             width_ths=0.7,
#             add_margin=0.1
#         )

#         texts = [self.process_text(text) for _, text, conf in results if conf >= self.confidence_threshold]
#         full_text = ' '.join(texts)

#         if full_text and full_text != self.last_spoken:
#             self.last_spoken = full_text
#             print("checking")
#             threading.Thread(target=self.speak_text, args=(full_text,)).start()
#             print("working")

#         return results

#     def run(self):
#         """Main capture loop"""
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Error capturing frame")
#                 break

#             # Process frame
#             results = self.process_frame(frame)

#             if results:
#                 for (bbox, text, _) in results:
#                     points = np.array(bbox).astype(np.int32)
#                     cv2.polylines(frame, [points], True, (0, 255, 0), 2)
#                     cv2.putText(frame, text, tuple(points[0]),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#             cv2.imshow('OCR System', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     ocr = MultiLangOCRSystem()
#     ocr.run()






















# import cv2
# import numpy as np
# import easyocr
# import re
# import time
# import threading
# import subprocess
# from collections import defaultdict

# class MultiLangOCRSystem:
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         self.languages = ['en', 'te']
#         self.reader = easyocr.Reader(self.languages, gpu=False)

#         # Updated voice mapping with proper eSpeak identifiers
#         self.voice_mapping = {
#             'ta': 'tam',
#             'hi': 'hin',
#             'te': 'tel',
#             'kn': 'kan',
#             'ml': 'mal',
#             'en': 'en'
#         }

#         self.script_ranges = {
#             'ta': (0x0B80, 0x0BFF),   # Tamil
#             'hi': (0x0900, 0x097F),   # Devanagari (Hindi)
#             'te': (0x0C00, 0x0C7F),   # Telugu
#             'kn': (0x0C80, 0x0CFF),   # Kannada
#             'ml': (0x0D00, 0x0D7F),   # Malayalam
#             'en': (0x0000, 0x007F)    # Basic Latin
#         }

#         self.text_buffer = []
#         self.last_spoken = ""
#         self.confidence_threshold = 0.4
#         self.process_interval = 0.5
#         self.last_process_time = 0
#         self.lock = threading.Lock()

#         print("Multi-language OCR System initialized!")

#     def optimize_image(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#         return cv2.medianBlur(enhanced, 3)

#     def detect_script(self, text):
#         script_counts = defaultdict(int)
#         for char in text:
#             code = ord(char)
#             for lang, (start, end) in self.script_ranges.items():
#                 if start <= code <= end:
#                     script_counts[lang] += 1
#                     break
#         return max(script_counts, key=script_counts.get) if script_counts else 'en'

#     def process_text(self, text):
#         text = re.sub(r'[^\w\s\-.,!?]', '', text)
#         return text.strip()

#     def speak_text(self, text):
#         lang = self.detect_script(text)
#         voice = self.voice_mapping.get(lang, 'en')

#         try:
#             # Modified eSpeak command with proper encoding
#             subprocess.run(
#                 ['espeak', '-v', voice, '-s', '150', '-p', '50', text],
#                 check=True,
#                 stderr=subprocess.DEVNULL  # Suppress error messages
#             )
#         except subprocess.CalledProcessError:
#             # Fallback to English if voice not found
#             if lang != 'en':
#                 print(f"Voice {voice} not found, falling back to English")
#                 subprocess.run(['espeak', '-v', 'en', text])
#         except Exception as e:
#             print(f"TTS Error: {str(e)}")

#     def process_frame(self, frame):
#         current_time = time.time()
#         if (current_time - self.last_process_time) < self.process_interval:
#             return

#         self.last_process_time = current_time
#         processed_img = self.optimize_image(frame)

#         results = self.reader.readtext(
#             processed_img,
#             text_threshold=0.6,
#             width_ths=0.7,
#             add_margin=0.1
#         )

#         texts = []
#         for (bbox, text, confidence) in results:
#             if confidence >= self.confidence_threshold:
#                 clean_text = self.process_text(text)
#                 if clean_text:
#                     texts.append(clean_text)

#         full_text = ' '.join(texts)
#         if full_text and full_text != self.last_spoken:
#             self.last_spoken = full_text
#             threading.Thread(target=self.speak_text, args=(full_text,)).start()

#         return results

#     def run(self):
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Error capturing frame")
#                 break

#             threading.Thread(target=self.process_frame, args=(frame.copy(),)).start()

#             results = self.process_frame(frame)
#             if results:
#                 for (bbox, text, _) in results:
#                     points = np.array(bbox).astype(np.int32)
#                     cv2.polylines(frame, [points], True, (0, 255, 0), 2)
#                     cv2.putText(frame, text, tuple(points[0]),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#             cv2.imshow('OCR System', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     ocr = MultiLangOCRSystem()
#     ocr.run()


# import cv2
# import numpy as np
# import easyocr
# import re
# import pyttsx3
# import time
# import threading
# import language_tool_python
# import os
# import langid
# from indic_transliteration import sanscript
# from spellchecker import SpellChecker

# class EnhancedWebcamOCR:
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         # Initialize OCR reader with optimized settings
#         self.reader = easyocr.Reader(['en','te'],
#                                     gpu=False,
#                                     quantize=True,
#                                     model_storage_directory='./models',
#                                     download_enabled=False)

#         # Language processing tools
#         self.spell_checker = SpellChecker()
#         self.grammar_tool = language_tool_python.LanguageTool('en-US')

#         # Text-to-speech engine configuration
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', 150)
#         self.tts_thread = None
#         self.tts_lock = threading.Lock()

#         # State management variables
#         self.last_spoken_text = ""
#         self.text_confidence_threshold = 0.6  # Increased confidence threshold
#         self.process_interval = 0.5
#         self.last_process_time = 0
#         self.processing_active = False

#         # Window configuration
#         cv2.namedWindow('Enhanced OCR', cv2.WINDOW_NORMAL)
#         print("OCR system initialized!")

#     def optimize_image(self, image):
#         """Enhanced image preprocessing pipeline for better OCR results"""
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Noise reduction using bilateral filter
#         denoised = cv2.bilateralFilter(gray, 9, 75, 75)

#         # Contrast Limited Adaptive Histogram Equalization
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(denoised)

#         # Adaptive thresholding for better text segmentation
#         thresholded = cv2.adaptiveThreshold(enhanced, 255,
#                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                            cv2.THRESH_BINARY, 11, 2)

#         return thresholded

#     def clean_text(self, text):
#         """Advanced text cleaning with regular expressions"""
#         # Remove special characters while preserving sentence structure
#         text = re.sub(r'[^\w\s.,!?:;()\-\'"À-ÿ]', '', text, flags=re.UNICODE)
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def apply_nlp(self, text):
#         """Enhanced NLP pipeline with context-aware corrections"""
#         # Language detection
#         lang, confidence = langid.classify(text)

#         # Only apply English corrections for now
#         if lang == 'en':
#             # Split text into words while preserving punctuation
#             words = re.findall(r'\w+[\'’]*\w*|[.,!?;]', text)

#             # Context-aware spell checking
#             corrected_words = []
#             for word in words:
#                 # Handle punctuation separately
#                 if word in {'.', ',', '!', '?', ';'}:
#                     corrected_words.append(word)
#                     continue

#                 # Get candidate corrections
#                 candidates = self.spell_checker.candidates(word.lower())
#                 if candidates:
#                     best_candidate = self.spell_checker.correction(word)
#                     # Preserve original capitalization
#                     if word.istitle():
#                         best_candidate = best_candidate.title()
#                     elif word.isupper():
#                         best_candidate = best_candidate.upper()
#                     corrected_words.append(best_candidate)
#                 else:
#                     corrected_words.append(word)

#             # Reconstruct text with proper spacing around punctuation
#             corrected_text = ' '.join(corrected_words)
#             corrected_text = re.sub(r'\s+([.,!?;])', r'\1', corrected_text)

#             # Grammar correction with error detection
#             matches = self.grammar_tool.check(corrected_text)
#             final_text = language_tool_python.utils.correct(corrected_text, matches)
#             return final_text

#         return text  # Return original text for non-English languages

#     def process_frame(self, frame):
#         """Optimized frame processing with accuracy improvements"""
#         self.processing_active = True
#         processed_image = self.optimize_image(frame)

#         # OCR with enhanced parameters
#         results = self.reader.readtext(processed_image,
#                                       text_threshold=0.7,
#                                       link_threshold=0.4,
#                                       width_ths=0.7,
#                                       decoder='greedy',
#                                       blocklist=None)

#         # Filter and process results
#         valid_results = []
#         for result in results:
#             bbox, text, prob = result
#             if prob >= self.text_confidence_threshold and text.strip():
#                 cleaned_text = self.clean_text(text)
#                 # Convert bbox to numpy array of integers
#                 np_bbox = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
#                 valid_results.append((np_bbox, cleaned_text, prob))

#         if valid_results:
#             # Sort results by vertical position using mean Y-coordinate
#             valid_results.sort(key=lambda x: np.mean(x[0][:, 0, 1]))

#             full_text = ' '.join([text for _, text, _ in valid_results])

#             if full_text != self.last_spoken_text:
#                 self.last_spoken_text = full_text
#                 nlp_text = self.apply_nlp(full_text)

#                 if nlp_text:
#                     threading.Thread(target=self.speak_text,
#                                    args=(nlp_text,),
#                                    daemon=True).start()

#         self.processing_active = False
#         return valid_results

#     def speak_text(self, text):
#         """Improved TTS with language detection"""
#         with self.tts_lock:
#             lang, _ = langid.classify(text)
#             try:
#                 # Set TTS voice based on detected language
#                 voices = self.engine.getProperty('voices')
#                 if lang == 'te':  # Telugu
#                     voice = next((v for v in voices if 'telugu' in v.name.lower()), voices[0])
#                 else:  # Default to English
#                     voice = next((v for v in voices if v.languages[0] == 'en'), voices[0])
#                 self.engine.setProperty('voice', voice.id)

#                 self.engine.say(text)
#                 self.engine.runAndWait()
#             except Exception as e:
#                 print(f"TTS Error: {str(e)}")

#     def run(self):
#         """Main loop with optimized frame processing"""
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Error: Failed to capture frame")
#                 break

#             current_time = time.time()
#             if not self.processing_active and (current_time - self.last_process_time) > self.process_interval:
#                 self.last_process_time = current_time
#                 results = self.process_frame(frame)

#                 # Draw bounding boxes and text
#                 for (bbox, text, prob) in results:
#                     # Draw polygon around text region
#                     cv2.polylines(frame, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
#                     # Get top-left corner for text placement
#                     top_left = tuple(bbox[0][0])
#                     cv2.putText(frame, f"{text} ({prob:.2f})", top_left,
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             cv2.imshow('Enhanced OCR', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     ocr = EnhancedWebcamOCR()
#     ocr.run()
