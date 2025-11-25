import cv2
import numpy as np
from pathlib import Path
from config.settings import config

class ModelTrainer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(config.HAAR_FILE)
    
    def prepare_training_data(self):
        """Prepare training data from known faces directory"""
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        for person_dir in config.KNOWN_DIR.iterdir():
            if not person_dir.is_dir():
                continue
                
            label_map[current_label] = person_dir.name
            for img_path in person_dir.glob("*.[jp][pn]g"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                detected = self.face_cascade.detectMultiScale(
                    img, scaleFactor=1.3, minNeighbors=5
                )
                for (x, y, w, h) in detected:
                    faces.append(img[y:y+h, x:x+w])
                    labels.append(current_label)
                    
            current_label += 1
        
        return faces, labels, label_map
    
    def train_model(self):
        """Train the LBPH recognizer with prepared data"""
        faces, labels, label_map = self.prepare_training_data()
        
        if not faces:
            print("❌ No faces found to train.")
            return None, None
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.write(str(config.TRAINER_FILE))
        np.save(config.LABELS_FILE, label_map)
        
        print("✔ Training completed!")
        return recognizer, label_map