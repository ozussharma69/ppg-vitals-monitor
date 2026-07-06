"""
download_data.py
Downloads and extracts the PPG-DaLiA dataset (Reiss et al., 2019) for
Project 6 - Wearable Vitals Monitor.

Source: University of Siegen (original authors' hosting)
https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html

The dataset (~2.7 GB zipped) contains per-subject pickle files (S1.pkl ... S15.pkl)
with synchronized PPG, 3-axis wrist accelerometer, chest ECG, respiration, and
activity labels for 15 subjects.
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Direct-download URL (ownCloud/sciebo share, "/download" suffix forces a raw stream)
DATA_URL = "https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "PPG_FieldStudy.zip")


def download_file(url: str, dest_path: str) -> None:
    """Stream-download a file with a progress bar, resuming isn't supported
    by this host so we just do a clean single-pass download."""
    print(f"Downloading PPG-DaLiA dataset from:\n  {url}\n")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with open(dest_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="PPG_FieldStudy.zip",
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"\nDownload complete: {dest_path}")


def extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"\nExtracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, extract_to)
    print("Extraction complete.")


def main():
    if os.path.exists(ZIP_PATH):
        print(f"Zip already exists at {ZIP_PATH}, skipping download.")
    else:
        try:
            download_file(DATA_URL, ZIP_PATH)
        except requests.exceptions.RequestException as e:
            print(f"\nDownload failed: {e}")
            print(
                "\nIf this keeps failing, the share link may have changed. "
                "Check https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html "
                "for the current download link and update DATA_URL in this script."
            )
            sys.exit(1)

    extract_zip(ZIP_PATH, DATA_DIR)

    print("\nDone. Expect a 'PPG_FieldStudy' folder inside data/ containing")
    print("S1/, S2/, ... S15/ subfolders, each with an SX.pkl file.")


if __name__ == "__main__":
    main()