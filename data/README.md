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
in YOLOv8 format, set your Roboflow API key and run:

```bash
export ROBOFLOW_API_KEY="your-api-key"
python utils/get_tiaia.py
```

In Windows PowerShell, use:

```bash
$env:ROBOFLOW_API_KEY="your-api-key"
python utils/get_tiaia.py
```

The dataset downloads to `data/tiaia/yolo/` in YOLO format (so we don't need to convert this time).