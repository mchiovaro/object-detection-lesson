# data/

Data are grouped by dataset:

```text
data/
├── aquarium/
│   ├── raw/       # Original COCO download
│   └── yolo/      # Converted images and YOLO labels
├── tiaia/
│   └── yolo/      # Roboflow v15 download; created by get_tiaia.py
└── reef_clip.mp4  # Existing tracking demo clip
```


## Part 1: Object Detection using Aquarium V2

To get the data, create the Python 3.12 environemnt with requirements.txt and run the following commands:
- `python utils/get_aquariumv2.py`
- `python utils/COCO_to_YOLO.py`


## Part 2: Object Tracking using Tiaia

To download [TIAIA Fish Species v4, version 15](https://universe.roboflow.com/reefos/tiaia-fish-species-v4/dataset/15/download)
in YOLOv8 format, follow the download instructions in the parent README.

We'll install **Roboflow SDK 1.7.3** in `.venv-download` (separate from the lesson's
`.venv`) to avoid overlapping OpenCV packages. 

The dataset downloads to `data/tiaia/yolo/` in YOLO format, so no conversion is needed.
After the downalod is complete, switch to the `.venv` from Part 1 of the lesson.
