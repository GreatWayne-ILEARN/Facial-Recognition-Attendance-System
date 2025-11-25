import cv2
import numpy as np
from pathlib import Path
from typing import List
from config.settings import config
from attendance.manager import AttendanceManager

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(config.HAAR_FILE)
        self.attendance_manager = AttendanceManager()
        self.recognizer = None
        self.label_map = None
        
    def load_model(self):
        """Load the trained model and label map"""
        if not config.TRAINER_FILE.exists():
            raise FileNotFoundError("Trainer not found. Please train the model first.")
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(str(config.TRAINER_FILE))
        self.label_map = np.load(config.LABELS_FILE, allow_pickle=True).item()
    
    def recognize_faces_in_image(self, image_path: Path, show_gui: bool = True):
        """Recognize faces in a single image"""
        if self.recognizer is None or self.label_map is None:
            self.load_model()
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # FIX: Convert NumPy array to boolean for the condition
        faces_detected = len(faces) > 0
        
        print(f"🔍 Found {len(faces)} face(s) in {image_path.name}")
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            id_, confidence = self.recognizer.predict(roi)
            name = self.label_map[id_] if confidence < 85 else "Unknown"
            
            print(f"👤 Recognized: {name} (confidence: {confidence:.2f})")
            
            # Mark attendance
            self.attendance_manager.mark_attendance(name)
            
            if show_gui:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # FIX: Use the boolean variable instead of the array
        if show_gui and faces_detected:
            cv2.imshow(f"Attendance - {image_path.name}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif show_gui and not faces_detected:
            print(f"❌ No faces detected in {image_path.name}")