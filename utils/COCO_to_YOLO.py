import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "aquarium_raw"
DESTINATION = ROOT / "data" / "aquarium_clean"

# established classes from the Aquarium Combined dataset
CLASSES = [
    "fish",
    "jellyfish",
    "penguin",
    "puffin",
    "shark",
    "starfish",
    "stingray",
]


def convert_split(annotation_file, split):
    """Convert COCO bounding boxes into the format YOLO expects.

    COCO describes each box using its top-left position and size:
        [x, y, width, height] in pixels.

    YOLO uses the center position and size:
        [center_x, center_y, width, height] as fractions of the image dimensions.

    We first clip the box to the image boundaries and skip boxes with no
    remaining width or height. Valid boxes inside the image stay unchanged.

    We then calculate the box's center and size, dividing horizontal values
    by the image width and vertical values by the image height. For example,
    a 200-pixel-wide box in a 400-pixel-wide image has a normalized width of 0.5.

    The conversion describes the same labeled object in a different format;
    it does not resize the image.
    """

    # load the annotation file for the current split
    coco = json.loads(annotation_file.read_text())

    # specify directory and make folders if they don't exist
    image_dir = DESTINATION / "images" / split
    label_dir = DESTINATION / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # make a dict, matching each COCO ID to its category name
    category_names = {
        category["id"]: category["name"]
        for category in coco["categories"]
    }

    # create a dictionary mapping each image ID to its annotations
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    # initialize counter for skipped boxes
    skipped_boxes = 0
    clipped_boxes = 0

    for image in coco["images"]:
        # find the actual image file corresponding to this annotation
        source_image = annotation_file.parent / image["file_name"]
        filename = Path(image["file_name"]).name

        # get the image dimensions from the COCO annotation to use for normalization
        width = image["width"]
        height = image["height"]

        rows = []

        # find that image's annotations from the dictionary, get the class, 
        # and process each annotation to convert it to YOLO format
        for annotation in annotations_by_image[image["id"]]:
            class_name = category_names[annotation["category_id"]]
            class_id = CLASSES.index(class_name)

            # COCO format is: top-left x, top-left y, width, height (in pixels)
            x, y, box_width, box_height = annotation["bbox"]

            # check and fix the box coordinates if they extend beyond the image boundaries
            x1 = max(0, min(width, x))
            y1 = max(0, min(height, y))
            x2 = max(0, min(width, x + box_width))
            y2 = max(0, min(height, y + box_height))

            # count how many boxes get clipped
            if x1 != x or y1 != y or x2 != x + box_width or y2 != y + box_height:
                clipped_boxes += 1

            # skip boxes that have no remaining width or height after clipping
            if x2 <= x1 or y2 <= y1:
                skipped_boxes += 1

                # display original coordinates to confirm why it was skipped
                print(f"\nImage: {filename}")
                print(f"Annotation ID: {annotation['id']}")
                print(f"Class: {class_name}")
                print(f"Original box [x, y, width, height]: {annotation['bbox']}")

                continue

            # YOLO: center x, center y, width, height (normalized)
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height

            # append the YOLO-formatted line to the list of rows for this image
            rows.append(
                f"{class_id} {center_x:.8f} {center_y:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}"
            )

        # copy the original image to the output directory
        shutil.copy2(source_image, image_dir / filename)

        # write the YOLO-formatted labels to a text file, one line per annotation
        label_file = label_dir / f"{Path(filename).stem}.txt"
        label_file.write_text("\n".join(rows))
        
    # print a summary of the conversion for this split
    print(
        f"{split}: {len(coco['images'])} images converted; "
        f"{clipped_boxes} boxes clipped; {skipped_boxes} invalid boxes skipped."
    )

def main():
    # Match each source folder to its output split name.
    for source_split, output_split in [
        ("train", "train"),
        ("valid", "val"),
        ("test", "test"),
    ]:
        annotation_file = SOURCE / source_split / "_annotations.coco.json"
        convert_split(annotation_file, output_split)


if __name__ == "__main__":
    main()
