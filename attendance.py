import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------- CONFIG ----------
ATTENDANCE_FILE = Path("attendance.csv")
KNOWN_DIR = Path("known_faces")
TRAINER_FILE = Path("trainer.yml")
LABELS_FILE = Path("labels.npy")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

KNOWN_DIR.mkdir(exist_ok=True)

# ---------- ATTENDANCE ----------
def has_marked_today(name: str) -> bool:
    if not ATTENDANCE_FILE.exists():
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATTENDANCE_FILE, "r") as f:
        for line in f:
            person, timestamp = line.strip().split(",")
            if person == name and timestamp.startswith(today):
                return True
    return False

def mark_attendance(name: str):
    if name == "Unknown" or has_marked_today(name):
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{now}\n")
    print(f"✔ Attendance marked for {name}")

# ---------- REGISTER NEW FACE ----------
def register_new_face():
    name = input("Enter the person's name: ").strip()
    person_dir = KNOWN_DIR / name
    person_dir.mkdir(exist_ok=True)

    path = input("Enter path to the image of the person: ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    img = cv2.imread(path)
    if img is None:
        print("❌ Invalid image.")
        return

    count = len(list(person_dir.iterdir())) + 1
    save_path = person_dir / f"{count}.jpg"
    cv2.imwrite(str(save_path), img)
    print(f"✔ Saved new face: {save_path}")

# ---------- TRAIN LBPH RECOGNIZER ----------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(HAAR_FILE)

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_dir in KNOWN_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        label_map[current_label] = person_dir.name
        for img_path in person_dir.glob("*.[jp][pn]g"):  # jpg, jpeg, png
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            detected = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in detected:
                faces.append(img[y:y+h, x:x+w])
                labels.append(current_label)
        current_label += 1

    if not faces:
        print("❌ No faces found to train.")
        return None, None

    recognizer.train(faces, np.array(labels))
    recognizer.write(str(TRAINER_FILE))
    np.save(LABELS_FILE, label_map)
    print("✔ Training completed!")
    return recognizer, label_map

# ---------- RECOGNIZE MULTIPLE IMAGES ----------
def recognize_images(image_paths, show_gui=True):
    if not TRAINER_FILE.exists():
        print("⚠ Trainer not found. Training now...")
        recognizer, label_map = train_model()
        if recognizer is None:
            return
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(TRAINER_FILE))
        label_map = np.load(LABELS_FILE, allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(HAAR_FILE)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"❌ Could not read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(f"🔍 Found {len(faces)} face(s) in {path}")

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi)
            name = label_map[id_] if confidence < 85 else "Unknown"
            mark_attendance(name)
            if show_gui:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if show_gui:
            cv2.imshow(f"Attendance - {Path(path).name}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# ---------- MAIN MENU ----------
def main():
    print("\n===== FACIAL RECOGNITION ATTENDANCE SYSTEM =====\n")
    print("1. Register a new face")
    print("2. Run attendance on image(s)")
    print("3. Train recognizer")
    choice = input("Enter choice: ").strip()

    if choice == "1":
        register_new_face()
    elif choice == "2":
        paths_input = input("Enter image paths separated by commas, or folder path: ").strip()
        p = Path(paths_input)
        if p.is_dir():
            image_paths = [str(f) for f in p.glob("*.[jp][pn]g")]
        else:
            image_paths = [s.strip() for s in paths_input.split(",")]
        recognize_images(image_paths)
    elif choice == "3":
        train_model()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
