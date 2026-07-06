\# Wearable Vitals Monitor — PPG-Based Heart Rate Estimation



Estimates heart rate (bpm) from wrist-worn PPG and accelerometer signals during daily-life activities, using a CNN+LSTM model trained on the \*\*PPG-DaLiA\*\* dataset. Motion artifacts are one of the core challenges in wearable HR sensing — this project explicitly models them via an accelerometer fusion branch rather than relying on PPG alone.



\## Overview



\- \*\*Dataset:\*\* \[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) (Reiss et al., 2019) — 15 subjects, 8 daily-life activities (sitting, stairs, table soccer, cycling, driving, lunch, walking, working), wrist PPG + 3-axis accelerometer, chest ECG ground truth.

\- \*\*Task:\*\* Regress heart rate (bpm) from an 8-second window of wrist PPG + accelerometer, sampled every 2 seconds.

\- \*\*Architecture:\*\* Dual 1D-CNN branches (PPG, accelerometer) → pooled to shared temporal length → concatenated → LSTM → FC regression head.

\- \*\*Evaluation protocol:\*\* Leave-one-subject-out (LOSO) cross-validation across all 15 subjects — the standard benchmark protocol for this dataset (matches the original Deep PPG paper).



\## Results



| Metric | Value |

|---|---|

| Mean MAE (all 15 subjects, LOSO) | \*\*8.63 bpm\*\* |

| Mean MAE (excluding 2 outlier subjects) | \*\*6.71 bpm\*\* |

| Best subject | S10 — 5.04 bpm |

| Worst subject | S5 — 23.27 bpm |



\*\*Finding:\*\* 13 of 15 subjects clustered tightly between 5–10 bpm MAE. Two subjects (S5, S8) showed substantially higher error (18–23 bpm), consistent with known high-motion-artifact recordings in this dataset (likely from intense cycling/exercise segments where PPG signal quality degrades independent of accelerometer compensation). This subject-dependent variance is itself a meaningful result — it highlights the real-world generalization challenge in wearable HR sensing, where a small number of "hard" users can dominate an averaged metric if not analyzed individually.



See `reports/loso\_results.csv` and `reports/loso\_activity\_mae.csv` for full per-subject and per-activity breakdowns.



\## Repository Structure



ppg-vitals-monitor/

├── src/

│   ├── download\_data.py   # Downloads PPG-DaLiA from original authors' hosting

│   ├── preprocess.py      # Windowing (8s/2s stride), bandpass filtering, PPG/ACC/label alignment

│   ├── train.py           # Model + dataset definitions, single-split training

│   └── evaluate.py        # Full leave-one-subject-out cross-validation

├── app/

│   └── streamlit\_app.py   # Interactive demo: pick a subject/window, see predicted vs true HR

├── reports/

│   ├── ppg\_hr\_model.pth       # Best trained model checkpoint

│   ├── loso\_results.csv       # Per-subject LOSO MAE

│   └── loso\_activity\_mae.csv  # Per-subject, per-activity MAE breakdown

└── data/                  # Raw + processed data (gitignored)



\## Setup



```bash

python -m venv venv

venv\\Scripts\\Activate.ps1   # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install numpy scipy pandas matplotlib scikit-learn requests tqdm streamlit

```



\## Usage



```bash

python src/download\_data.py     # download + extract PPG-DaLiA (\~2.7 GB)

python src/preprocess.py        # window + filter all 15 subjects

python src/train.py             # train on a fixed split (fast sanity check)

python src/evaluate.py          # full leave-one-subject-out cross-validation

streamlit run app/streamlit\_app.py   # interactive demo

```



\## Why this project



Wrist-worn PPG sensors are cheap and convenient but highly sensitive to motion artifacts — a real constraint for continuous vitals monitoring in consumer and clinical wearables alike. This project demonstrates: (1) building a full data pipeline from a real physiological dataset, (2) a sensor-fusion deep learning approach to a known hard problem, (3) rigorous subject-independent evaluation rather than reporting an optimistic single split, and (4) honest characterization of failure modes rather than hiding them.



\## Citation



Reiss, A., Indlekofer, I., Schmidt, P., \& Van Laerhoven, K. (2019). Deep PPG: Large-scale Heart Rate Estimation with Convolutional Neural Networks. \*MDPI Sensors\*, 19(14). https://doi.org/10.3390/s19143079

