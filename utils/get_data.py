from pathlib import Path

import kagglehub


def download_data():
    # re-direct the download to the local data directory
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "aquarium_raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        "kushagrasharma133/aquarium-combined-v2-raw-1024-coco-mmdetection",
        output_dir=str(data_dir),
    )

    print(f"Dataset downloaded to: {downloaded_path}")
    return Path(downloaded_path)


if __name__ == "__main__":
    download_data()