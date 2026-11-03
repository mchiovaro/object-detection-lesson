from pathlib import Path
import numpy as np
from PIL import Image
from utils.COCO_to_YOLO import CLASSES

DATA_DIR = Path(__file__).parent.parent / "data" / "aquarium_clean"

def read_labels(image_path):
    """Read YOLO labels and convert normalized boxes back to pixel corners.

    Returns one row per object: [class_id, xmin, ymin, xmax, ymax].
    This reverses the coordinate conversion we used in reformat_labels.py.
    """
    image_path = Path(image_path)
    split = image_path.parent.name
    label_file = DATA_DIR / "labels" / split / f"{image_path.stem}.txt"
    with Image.open(image_path) as image:
        width, height = image.size

    boxes = []
    for row in label_file.read_text().splitlines():
        if not row.strip():
            continue
        class_id, cx, cy, box_width, box_height = map(float, row.split())

        # check IDs and coordinates before using them in a plot or metric
        values = [class_id, cx, cy, box_width, box_height]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite label in {label_file}")
        if class_id != int(class_id) or not 0 <= class_id < len(CLASSES):
            raise ValueError(f"Invalid class ID in {label_file}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            raise ValueError(f"Invalid normalized box in {label_file}")

        # convert normalized coordinates to pixel coordinates
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height

        # allow a tiny tolerance because labels were rounded to eight decimals
        if x1 < -0.001 or y1 < -0.001 or x2 > width + 0.001 or y2 > height + 0.001:
            raise ValueError(f"Box extends outside the image in {label_file}")
        boxes.append([int(class_id), x1, y1, x2, y2])

    return np.asarray(boxes, dtype=float).reshape(-1, 5)