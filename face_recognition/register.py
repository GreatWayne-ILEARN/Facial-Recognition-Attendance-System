import cv2
import os
from pathlib import Path
from config.settings import config

class FaceRegister:
    def __init__(self, known_dir: Path = None):
        self.known_dir = known_dir or config.KNOWN_DIR
    
    def register_new_face(self) -> bool:
        """Register a new face from an image file"""
        name = input("Enter the person's name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return False
        
        person_dir = self.known_dir / name
        person_dir.mkdir(exist_ok=True)
        
        path = input("Enter path to the image of the person: ").strip()
        if not os.path.exists(path):
            print("❌ File not found.")
            return False
        
        img = cv2.imread(path)
        if img is None:
            print("❌ Invalid image.")
            return False
        
        count = len(list(person_dir.iterdir())) + 1
        save_path = person_dir / f"{count}.jpg"
        cv2.imwrite(str(save_path), img)
        print(f"✔ Saved new face: {save_path}")
        return True