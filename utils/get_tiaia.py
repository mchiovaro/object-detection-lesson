"""Download TIAIA Fish Species v4, version 15, from Roboflow Universe.

Set ROBOFLOW_API_KEY in your environment before running this script.
Source: https://universe.roboflow.com/reefos/tiaia-fish-species-v4/dataset/15
"""

import os
from pathlib import Path


def download_data():

    # get the api key and flag if it is not set
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY before running."
        )

    # import roboflow; if it fails, raise exit with a message
    try:
        from roboflow import Roboflow
    except ModuleNotFoundError as exc:
        if exc.name != "roboflow":
            raise
        raise SystemExit(
            "Install the roboflow dependency roboflow"
        ) from exc

    # create the data directory if it does not exist
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "tiaia" / "yolo"
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    # download the dataset from Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("reefos").project("tiaia-fish-species-v4")
    dataset = project.version(15).download("yolov8", location=str(data_dir))

    # print the path to the downloaded dataset and print it
    downloaded_path = Path(dataset.location)
    print(f"Dataset downloaded to: {downloaded_path}")
    return downloaded_path


if __name__ == "__main__":
    download_data()
