"""
streamlit_app.py
Interactive demo for the PPG-based heart rate estimation model.

Lets you pick a subject and a specific window, see the raw PPG + accelerometer
signal, and compare the model's predicted HR against the ground-truth HR
(derived from chest ECG).

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

# Make src/ importable so we can reuse the exact same model + dataset code
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)
from train import PPGHRNet, DEVICE  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "ppg_hr_model.pth")

st.set_page_config(page_title="PPG Wearable Vitals Monitor", layout="wide")


@st.cache_resource
def load_model():
    model = PPGHRNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


@st.cache_data
def load_subject_data(subject_id):
    data = np.load(os.path.join(DATA_DIR, f"{subject_id}.npz"))
    return data["ppg"], data["acc"], data["hr_label"], data["activity"]


def predict_hr(model, ppg_window, acc_window):
    """Runs a single window through the model, applying the same
    per-window z-score normalization used during training."""
    ppg_norm = (ppg_window - ppg_window.mean()) / (ppg_window.std() + 1e-8)
    acc_norm = (acc_window - acc_window.mean(axis=0)) / (acc_window.std(axis=0) + 1e-8)

    ppg_tensor = torch.tensor(ppg_norm, dtype=torch.float32).view(1, 1, -1).to(DEVICE)
    acc_tensor = torch.tensor(acc_norm, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(ppg_tensor, acc_tensor).item()
    return pred


def main():
    st.title("Wearable Vitals Monitor")
    st.caption("PPG + accelerometer -> heart rate estimation, trained on PPG-DaLiA (CNN + LSTM)")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model checkpoint not found at {MODEL_PATH}. Run src/train.py first.")
        return

    model = load_model()

    # Subjects S14/S15 were held out during training in train.py, so predictions
    # on them are genuinely out-of-sample -- an honest demo, not a memorized result.
    available_subjects = [f"S{i}" for i in range(1, 16)]

    with st.sidebar:
        st.header("Controls")
        subject_id = st.selectbox(
            "Subject",
            available_subjects,
            index=13,
            help="S14 and S15 were held out during training (in train.py) -- predictions on "
                 "them are genuinely out-of-sample. Other subjects were seen during training.",
        )
        ppg, acc, hr_label, activity = load_subject_data(subject_id)
        window_idx = st.slider("Window index", 0, len(hr_label) - 1, len(hr_label) // 2)

        if subject_id not in ("S14", "S15"):
            st.info("This subject's data was used during training -- prediction will likely look better than real-world performance. Pick S14 or S15 for an honest out-of-sample demo.")

    ppg_window = ppg[window_idx]
    acc_window = acc[window_idx]
    true_hr = hr_label[window_idx]
    activity_id = activity[window_idx]

    pred_hr = predict_hr(model, ppg_window, acc_window)
    error = abs(pred_hr - true_hr)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted HR", f"{pred_hr:.1f} bpm")
    col2.metric("Ground Truth HR (ECG)", f"{true_hr:.1f} bpm")
    col3.metric("Absolute Error", f"{error:.1f} bpm")

    st.caption(f"Activity ID: {activity_id}  |  Window: {window_idx + 1} / {len(hr_label)}  |  8-second window, 2-second stride")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)

    time_ppg = np.arange(len(ppg_window)) / 64.0  # 64 Hz
    ax1.plot(time_ppg, ppg_window, color="#d62728")
    ax1.set_title("Wrist PPG (bandpass filtered)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    time_acc = np.arange(len(acc_window)) / 32.0  # 32 Hz
    ax2.plot(time_acc, acc_window[:, 0], label="X", alpha=0.8)
    ax2.plot(time_acc, acc_window[:, 1], label="Y", alpha=0.8)
    ax2.plot(time_acc, acc_window[:, 2], label="Z", alpha=0.8)
    ax2.set_title("Wrist Accelerometer (3-axis)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("About this model")
    st.markdown(
        """
        - **Architecture:** Dual 1D-CNN branches (PPG + accelerometer) fused and fed into an LSTM, regressing to a single HR value.
        - **Training:** PPG-DaLiA dataset, 15 subjects, 8 daily-life activities (sitting, cycling, walking, stairs, driving, etc.)
        - **Evaluation:** Leave-one-subject-out cross-validation across all 15 subjects gave a mean MAE of **8.63 bpm** (13 of 15 subjects clustered between 5-10 bpm; 2 subjects with known high-motion-artifact recordings showed 18-23 bpm MAE).
        - **Why it matters:** Wrist PPG is corrupted by motion artifacts during exercise. The accelerometer branch lets the model learn to compensate, rather than relying on PPG alone.
        """
    )


if __name__ == "__main__":
    main()