"""
preprocess.py
Windows, filters, and aligns PPG-DaLiA signals for HR estimation.

Standard protocol (matches the Deep PPG paper and most follow-up work):
- 8-second windows, 2-second stride (i.e. label matches the wrist sampling of HR every 2s)
- Wrist PPG @ 64 Hz  -> 512 samples/window
- Wrist ACC @ 32 Hz  -> 256 samples/window (3 axes)
- Label (HR, ground truth from chest ECG) is already provided at 0.5 Hz (one value per 2s)
  so each label value corresponds 1:1 to one window's centered/trailing 8s of signal.

Output: one .npz file per subject in data/processed/, containing:
    ppg      : (N, 512)      windowed, bandpass-filtered PPG
    acc      : (N, 256, 3)   windowed wrist accelerometer (x,y,z)
    hr_label : (N,)          ground-truth HR in bpm for each window
    activity : (N,)          activity ID for each window (for per-activity evaluation)
    subject  : str
"""

import os
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "PPG_FieldStudy")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

PPG_FS = 64          # wrist PPG sampling rate (Hz)
ACC_FS = 32          # wrist accelerometer sampling rate (Hz)
WINDOW_SEC = 8        # window length in seconds
STRIDE_SEC = 2         # stride in seconds (matches label rate of 0.5 Hz)

PPG_WINDOW_SAMPLES = WINDOW_SEC * PPG_FS   # 512
ACC_WINDOW_SAMPLES = WINDOW_SEC * ACC_FS   # 256
PPG_STRIDE_SAMPLES = STRIDE_SEC * PPG_FS   # 128
ACC_STRIDE_SAMPLES = STRIDE_SEC * ACC_FS   # 64


def bandpass_filter(signal: np.ndarray, fs: int, low=0.5, high=4.0, order=4) -> np.ndarray:
    """Bandpass filter for PPG: 0.5-4 Hz corresponds to 30-240 bpm, covering
    resting HR through vigorous exercise (cycling/stair activities in this dataset)."""
    nyquist = fs / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal, axis=0)


def load_subject(subject_id: str) -> dict:
    path = os.path.join(RAW_DIR, subject_id, f"{subject_id}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def window_subject(data: dict, subject_id: str) -> dict:
    ppg_raw = data["signal"]["wrist"]["BVP"].squeeze()      # PPG channel is called BVP (blood volume pulse)
    acc_raw = data["signal"]["wrist"]["ACC"]                # shape (T, 3)
    hr_label = np.asarray(data["label"]).squeeze()           # HR every 2s
    activity_raw = np.asarray(data["activity"]).squeeze()

    # Bandpass filter the full PPG signal before windowing (filtfilt needs continuous signal)
    ppg_filtered = bandpass_filter(ppg_raw, fs=PPG_FS)

    n_windows = len(hr_label)  # one label per 2s window, this defines how many windows we can make

    ppg_windows = []
    acc_windows = []
    activity_windows = []
    valid_labels = []

    for i in range(n_windows):
        ppg_end = (i + 1) * PPG_STRIDE_SAMPLES
        ppg_start = ppg_end - PPG_WINDOW_SAMPLES
        acc_end = (i + 1) * ACC_STRIDE_SAMPLES
        acc_start = acc_end - ACC_WINDOW_SAMPLES

        # Skip windows before enough history has accumulated (start of recording)
        if ppg_start < 0 or acc_start < 0:
            continue
        if ppg_end > len(ppg_filtered) or acc_end > len(acc_raw):
            continue

        ppg_windows.append(ppg_filtered[ppg_start:ppg_end])
        acc_windows.append(acc_raw[acc_start:acc_end])
        activity_windows.append(activity_raw[i] if i < len(activity_raw) else -1)
        valid_labels.append(hr_label[i])

    return {
        "ppg": np.array(ppg_windows, dtype=np.float32),
        "acc": np.array(acc_windows, dtype=np.float32),
        "hr_label": np.array(valid_labels, dtype=np.float32),
        "activity": np.array(activity_windows, dtype=np.int32),
        "subject": subject_id,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subject_ids = [f"S{i}" for i in range(1, 16)]

    for subject_id in tqdm(subject_ids, desc="Preprocessing subjects"):
        raw = load_subject(subject_id)
        processed = window_subject(raw, subject_id)

        out_path = os.path.join(OUT_DIR, f"{subject_id}.npz")
        np.savez_compressed(
            out_path,
            ppg=processed["ppg"],
            acc=processed["acc"],
            hr_label=processed["hr_label"],
            activity=processed["activity"],
        )
        print(
            f"  {subject_id}: {processed['ppg'].shape[0]} windows, "
            f"HR range [{processed['hr_label'].min():.1f}, {processed['hr_label'].max():.1f}] bpm"
        )

    print(f"\nAll subjects processed. Output saved to {OUT_DIR}")


if __name__ == "__main__":
    main()