# Sources and attribution

## Aquarium Combined

Aquarium Combined v2 raw-1024 dataset by Roboflow (CC BY 4.0):
https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2

The original split contains 448 training, 127 validation and 63 test images with seven annotated categories. The data were obtained from the GLIP authors' public mirror, pinned at revision `66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1`:
https://huggingface.co/GLIPModel/GLIP/tree/66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1/odinw/Aquarium

Changes for this lesson: convert COCO annotations to normalized YOLO boxes, map the seven annotated category IDs to 0–6, ignore the unused parent category, preserve image aspect ratio while reducing longest side to at most 640 pixels, and re-encode JPEG quality 85. Two degenerate boxes were removed. The original train/validation/test split is preserved and all 448 training images are used. The full preparation scripts are in `utils/`.

## Ultralytics

Starting nano weights: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt