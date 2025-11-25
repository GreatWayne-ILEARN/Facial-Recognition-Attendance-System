from face_recognition.register import FaceRegister
from face_recognition.trainer import ModelTrainer
from face_recognition.recognizer import FaceRecognizer
from utils.helpers import parse_image_paths

def main():
    print("\n===== FACIAL RECOGNITION ATTENDANCE SYSTEM =====\n")
    print("1. Register a new face")
    print("2. Run attendance on image(s)")
    print("3. Train recognizer")
    
    choice = input("Enter choice: ").strip()
    
    if choice == "1":
        register = FaceRegister()
        register.register_new_face()
    elif choice == "2":
        paths_input = input("Enter image paths separated by commas, or folder path: ").strip()
        image_paths = parse_image_paths(paths_input)
        
        recognizer = FaceRecognizer()
        for image_path in image_paths:
            recognizer.recognize_faces_in_image(image_path)
    elif choice == "3":
        trainer = ModelTrainer()
        trainer.train_model()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()