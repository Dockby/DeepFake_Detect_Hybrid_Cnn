# DeepFake Detection — Hybrid CNN-Attention Model

> **Paper:** DeepFake Detection Using Hybrid CNN Attention Model  
> **Authors:** Chetan Singh · Hardik Garg  
> **Guide:** Dr. Swati Sharma  
> **Institution:** Galgotias University, Greater Noida, India

---

## 🚀 Quick Start (Run in 3 steps)

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/deepfake-detect.git
cd deepfake-detect
pip install -r requirements.txt

# 2. Create sample inputs
python create_sample_inputs.py

# 3. Run demo  (no dataset needed!)
python demo.py
```

Results saved to `results/demo_output.png`

---

## 📊 Paper Results on DFD Test Set

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| MesoNet | 88.2% | 86.4% | 88.7% | 87.5% | 0.891 |
| VGGNet | 91.5% | 90.2% | 91.0% | 90.6% | 0.921 |
| Xception | 93.1% | 93.3% | 93.2% | 93.2% | 0.944 |
| **Ours (Proposed)** | **95.5%** | **95.2%** | **95.7%** | **95.4%** | **0.971** |

### Ablation Study

| Variant | Accuracy | F1 |
|---|---|---|
| CNN Only (Xception) | 91.8% | 91.2% |
| + Channel Attention | 93.4% | 92.9% |
| + Spatial Attention | 94.1% | 93.8% |
| **+ Both (Proposed)** | **95.5%** | **95.4%** |

---

## 🏗️ Architecture

```
Input (224 × 224 × 3)
        ↓
Xception Backbone (ImageNet pretrained, fine-tuned)
        ↓
Feature Maps  7 × 7 × 2048
    ┌─────────────────────────────┐
    │  Channel Attention          │  → suppress uninformative channels
    │  Spatial Attention          │  → focus on eyes, mouth, hairline
    └─────────────────────────────┘
        ↓
Global Average Pooling  →  2048
        ↓
Dense 512 → Dropout 0.5 → Sigmoid
        ↓
   REAL  /  FAKE
```

**Why sigmoid attention (not softmax)?**  
Sigmoid lets the model attend to *multiple facial regions independently* — eyes, mouth boundaries, and hairline can all be activated at the same time. Softmax would force competition between regions, diluting the signal.

---

## 📁 File Structure

```
deepfake-detect/
│
├── model.py                 ← Model architecture
├── preprocess.py            ← Frame extraction + MTCNN face detection
├── train.py                 ← Training script
├── evaluate.py              ← Metrics, plots, Grad-CAM heatmaps
├── inference.py             ← Predict on any image or video
├── demo.py                  ← Quick demo (no dataset needed)
├── create_sample_inputs.py  ← Generate synthetic test images
├── kaggle_notebook.py       ← Complete Kaggle end-to-end script
├── requirements.txt
│
├── sample_inputs/           ← Created by create_sample_inputs.py
│   ├── test_real_01.jpg
│   ├── test_real_02.jpg
│   ├── test_real_03.jpg
│   ├── test_fake_01.jpg
│   ├── test_fake_02.jpg
│   └── test_fake_03.jpg
│
├── results/                 ← Outputs (created automatically)
│   ├── demo_output.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   └── metrics.txt
│
├── saved_models/            ← Trained weights go here
│   ├── best_model.h5
│   └── final_model.h5
│
└── logs/
    └── training_log.csv
```

---

## 📦 Dataset (DFD)

Download from Kaggle:  
🔗 https://www.kaggle.com/datasets/soroush365/deep-fake-detection-dfd-entire-original-dataset

Place videos as:
```
data/
└── raw/
    ├── real/    ← original face videos (.mp4)
    └── fake/    ← deepfake videos (.mp4)
```

| Split | Real | Fake | Total | Videos |
|---|---|---|---|---|
| Train | 6,985 | 6,915 | 13,900 | ~300 |
| Val | 1,500 | 1,485 | 2,985 | ~65 |
| Test | 915 | 885 | 1,800 | ~40 |
| **Total** | **9,400** | **9,285** | **18,685** | **~405** |

---

## 🔧 Full Training Pipeline

```bash
# Step 1: Preprocess (extract face crops from videos)
python preprocess.py --raw data/raw --out data/processed

# Step 2: Train (20 epochs, Adam, lr=1e-4, batch=32)
python train.py --data data/processed

# Step 3: Evaluate on test set
python evaluate.py --model saved_models/best_model.h5 --data data/processed

# Step 4: Predict on any image or video
python inference.py --input path/to/image.jpg
python inference.py --input path/to/video.mp4 --heatmap
```

---

## 🖥️ Running on Kaggle

1. Open `kaggle_notebook.py` — it is structured as a self-contained Kaggle script
2. In your Kaggle notebook, go to **Settings → Add Data** and add the DFD dataset
3. Copy each cell into a Kaggle notebook cell and run sequentially
4. GPU T4 × 2 is recommended (matches paper hardware)

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Framework | TensorFlow 2.13 |
| GPU | NVIDIA T4 16 GB |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| LR scheduler | Halve after 3 epochs plateau |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | Patience = 5 |
| Convergence | Epoch 14–18 |
| Dropout | 0.5 |
| Input size | 224 × 224 × 3 |

---

## 📧 Contact

| Name | Email |
|---|---|
| Chetan Singh | chetansingh1274@gmail.com |
| Hardik Garg | garghardik743@gmail.com |
| Dr. Swati Sharma (Guide) | swatisharma@galgotiasuniversity.edu.in |

---

## 📚 Citation

```bibtex
@inproceedings{singh2025deepfake,
  title     = {DeepFake Detection Using Hybrid CNN Attention Model},
  author    = {Singh, Chetan and Garg, Hardik and Sharma, Swati},
  booktitle = {IEEE Conference},
  year      = {2025},
  institution = {Galgotias University}
}
```
