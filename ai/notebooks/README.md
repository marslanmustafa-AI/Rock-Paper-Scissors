# 🧠 Rock-Paper-Scissors — Deep Learning Notebooks

> **A comparative study of three deep learning approaches for image classification.**

---

## 📖 Project Overview

This project trains and compares **three different deep learning models** to classify images of hand gestures (Rock, Paper, Scissors):

| # | Approach | Framework | Notebook |
|---|----------|-----------|----------|
| 0 | **Data Preparation** | Python (stdlib) | `data_split.ipynb` |
| 1 | **CNN from Scratch** | TensorFlow / Keras | `from_scratch/dl-v1.ipynb` |
| 2 | **YOLO Classification** | Ultralytics YOLO | `yolo/dl-v1.ipynb` |
| 3 | **Transfer Learning + Fine-Tuning** | TensorFlow / Keras (MobileNetV2) | `mobilenet/dl-v3-Transfer_learning+fine_tuning.ipynb` |

---

## 📁 Directory Structure

```
notebooks/
├── README.md                          ← You are here
├── data_split.ipynb                   ← Step 0: Split raw data into train/val/test
│
├── data/                              ← Raw dataset (rock, paper, scissors)
├── split_data/                        ← Output of data_split.ipynb
│   ├── train/
│   ├── val/
│   └── test/
│
├── from_scratch/                      ← Approach 1: Custom CNN
│   └── dl-v1.ipynb
│
├── yolo/                              ← Approach 2: YOLO classification
│   └── dl-v1.ipynb
│
├── mobilenet/                         ← Approach 3: MobileNetV2
│   └── dl-v3-Transfer_learning+fine_tuning.ipynb
│
├── weights/                           ← Trained model weights
│   ├── scratch/                       ← Custom CNN weights
│   ├── yolo/                          ← YOLO weights
│   └── mobilenet/
│       ├── transferlearning/          ← Phase 1 weights
│       └── finetune/                  ← Phase 2 weights
│
├── comparison/                        ← Model comparison notebooks
├── images/                            ← Supporting images
└── models/                            ← Legacy models (deprecated)
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| TensorFlow / Keras | ≥ 2.13 / Keras 3 |
| Ultralytics | Latest |
| Matplotlib | Latest |

### Run Order

1. **`data_split.ipynb`** — Splits the raw `data/` into `split_data/train`, `split_data/val`, `split_data/test` with automatic ratio selection based on dataset size.
2. **Any of the 3 model notebooks** — Each reads from `split_data/` and saves weights to `weights/`.
3. **`comparison/`** — (Optional) Compare model performance side by side.

---

## 🔬 Approach Comparison

| Feature | CNN from Scratch | YOLO | MobileNetV2 |
|---------|:---:|:---:|:---:|
| Architecture | Custom 3-block CNN | YOLOv26n-cls | MobileNetV2 (ImageNet) |
| Pre-trained | ❌ | ✅ | ✅ |
| Training strategy | End-to-end | Fine-tune all | Transfer → Fine-tune |
| Augmentation | Keras layers | Built-in YOLO | Keras layers |
| Best for | Learning fundamentals | Speed + accuracy | Production quality |

---

## 📝 Notes for Students

- **Always run `data_split.ipynb` first** to generate the train/val/test split.
- Each notebook is **self-contained** — run all cells top-to-bottom.
- Models are saved to `weights/` — you can load them later for inference.
- The `show_batch()` function in each notebook lets you visually inspect the data.
- Augmentation previews help you understand what transformations are applied.

---

## 👤 Author

**Muhammad Arslan**
