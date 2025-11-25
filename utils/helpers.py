from pathlib import Path
from typing import List

def parse_image_paths(paths_input: str) -> List[Path]:
    """Parse user input into list of image paths"""
    p = Path(paths_input.strip())
    
    if p.is_dir():
        # Get all images from directory
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(p.glob(ext))
        return image_paths
    else:
        # Split comma-separated paths
        return [Path(s.strip()) for s in paths_input.split(",")]