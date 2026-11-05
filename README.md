# Object Detection with Aquarium Images

A guided object detection (and later, object tracking) lesson for **EGR 444: Advanced AI/ML for Engineering Applications** at the University of Rhode Island.

In this lesson, we use **YOLOv8 nano** to locate and classify aquarium animals. We will go through the full workflow: inspecting the data, preparing bbox labels, fine-tuning a pretrained model, evaluatigng predictions, and investigating mistakes.

The goal is to understand both **how to train a detector** and **how to decide whether its predictions are useful**.

## What you will learn

- Distinguish classification, detection, segmentation, and tracking.
- Read COCO annotations and convert them into YOLO labels.
- Inspect images, bounding boxes, dataset splits, and class balance.
- Fine-tune a pretrained YOLO detector.
- Interpret training losses, precision–recall curves, per-class AP, and mAP at different IoU thresholds.
- Separate confidence thresholds, NMS overlap, and evaluation IoU.
- Examine missed objects, class confusion, and small-object performance.

## Repository contents

| File or folder | Purpose |
|---|---|
| `01_object_detection_lesson.ipynb` | Main lesson, including training, plots, evaluation, and exercises |
| `utils/get_data.py` | Downloads the original dataset |
| `utils/COCO_to_YOLO.py` | Converts COCO annotations into YOLO labels and copies images into split folders |
| `utils/YOLO_to_COCO.py` | Converts YOLO annotations into COCO labels for easier visualization/plotting |
| `requirements.txt` | Dependencies |
| `configs/train.yaml` | Baseline training settings |
| `models/` | Starting weights and any trained checkpoints |
| `outputs/` | Training results, checkpoints, plots, and evaluation tables |

Some folders (e.g., `outputs/`) are created when you run the notebook. 

## Dataset

For practice data, we use **Aquarium Combined v2**, which has seven classes:

| Class ID | Class |
|---:|---|
| 0 | fish |
| 1 | jellyfish |
| 2 | penguin |
| 3 | puffin |
| 4 | shark |
| 5 | starfish |
| 6 | stingray |

The provided split contains **448 training images, 127 validation images, and 63 test images**. We preserve that split for now, but we do check for corrupted/out-of-bounds bboxes and elimate them from individual examples.

The preparation script, `COCO_to_YOLO.py`, does the following:

1. Reads the original COCO annotations.
2. Maps category names into the class order above.
3. Clips boxes to the image boundaries.
4. Skips boxes with no remaining width or height.
5. Writes normalized YOLO labels and copies the original images to a clean location.

Note: This does not resize the images. YOLO handles resizing during training and prediction.

## Run locally

Use **Python 3.12**. Save the project wherever you prefer on your machine, then open a terminal in that folder.

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `01_object_detection_lesson.ipynb` in VS Code and select the `.venv` environment as its kernel.

In the first cell, set `LOCAL_ROOT` to your project folder. Run the notebook from the beginning.

## Run in Google Colab

1. Ensure all project files are inside a root folder named `object-detection-lesson`.
2. Zip that folder as `object-detection-lesson.zip`.
3. Upload `01_object_detection_lesson.ipynb` to Colab.
4. Run the first cell and upload the ZIP when prompted.
5. Run the dependency-installation cell. Restart the runtime if requested, then rerun from the beginning.

A GPU runtime can speed up training. Switch it at the beginning of the runtime so you don't lose work!

## Prepare the data

The notebook checks for prepared data and runs the preparation scripts when needed. You can also run them yourself from the project folder:

```bash
python utils/get_data.py
python utils/COCO_to_YOLO.py
```

The raw dataset (in `data/aquarium_raw/`) should directly contain `train/`, `valid/`, and `test/`. The conversion script renames the output validation split to `val/` and places the three split folders into `data/aquarium_clean/`.

Each prepared image has a matching text file with one row per object, in YOLO format:

```text
class_id center_x center_y width height
```

Note: the coordinates are fractions of the image dimensions.

## Train and inspect the model

Run the notebook cells in order. The starting example uses YOLOv8 nano, five epochs, input size 320, and all training images. Edit `configs/train.yaml` to change training settings.

The original pretrained model has COCO classes, and fine-tuning adapts it to the seven aquarium classes.

Training saves:

- `best.pt`: the checkpoint selected using validation performance.
- `last.pt`: the final training checkpoint.
- Training history, plots (from Ultralytics), and the configuration used.

If a trained aquarium checkpoint is available and you just want to evaluate, set `TRAIN_NOW = False` and point `REFERENCE_CHECKPOINT` to it. Otherwise, leave training enabled.

## Evaluate and experiment

The notebook examines:

- mAP50, mAP75, and mAP50–95
- Per-class AP and annotation counts
- Precision–recall curves
- Precision, recall, and F1 at explicit confidence thresholds
- A detection confusion matrix
- Individual mistakes and recall by object size
- NMS behavior, prediction latency, and model size

For the exercise, change **one training setting**, keep the other conditions fixed, and compare both metrics and example predictions.

Keep `RUN_FINAL_TEST = False` while choosing settings. Use the test split after your model and evaluation choices are fixed.

## Limits and next steps

These are aquarium photographs, not a tracking benchmark. The dataset does not provide identities across video frames. (For my EGR444 students, we grab a more appropriate dataset and look at tracking in a separate lab later in the semester.)

The supplied split also does not establish independence by exhibit or recording. Good aquarium validation performance does not demonstrate that a model will generalize to new images.

## Sources

- [Aquarium Combined v2 on Roboflow](https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2)
- [Kaggle dataset used by the download script](https://www.kaggle.com/datasets/kushagrasharma133/aquarium-combined-v2-raw-1024-coco-mmdetection)
- [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/)
- [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/)