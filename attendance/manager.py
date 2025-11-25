from datetime import datetime
from pathlib import Path
from config.settings import config

class AttendanceManager:
    def __init__(self, attendance_file: Path = None):
        self.attendance_file = attendance_file or config.ATTENDANCE_FILE
    
    def has_marked_today(self, name: str) -> bool:
        """Check if a person has already marked attendance today"""
        if not self.attendance_file.exists():
            return False
            
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with open(self.attendance_file, "r") as f:
                for line in f:
                    if line.strip():
                        person, timestamp = line.strip().split(",")
                        if person == name and timestamp.startswith(today):
                            return True
        except FileNotFoundError:
            pass
        return False
    
    def mark_attendance(self, name: str) -> bool:
        """Mark attendance for a person if not already marked today"""
        if name == "Unknown" or self.has_marked_today(name):
            return False
            
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.attendance_file, "a") as f:
            f.write(f"{name},{now}\n")
        print(f"✔ Attendance marked for {name}")
        return True