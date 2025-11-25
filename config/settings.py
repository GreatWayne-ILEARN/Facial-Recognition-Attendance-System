from pathlib import Path
import cv2

class Config:
    """Configuration settings for the application"""
    def __init__(self):
        self.ATTENDANCE_FILE = Path("attendance.csv")
        self.KNOWN_DIR = Path("known_faces")
        self.TRAINER_FILE = Path("trainer.yml")
        self.LABELS_FILE = Path("labels.npy")
        self.HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        # Create directories
        self.KNOWN_DIR.mkdir(exist_ok=True)

config = Config()