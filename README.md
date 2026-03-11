# DeepFake Detection — CNN + Attention

Implementation for our paper **"DeepFake Detection Using Hybrid CNN Attention Model"** submitted to IEEE.  
Authors: Chetan Singh, Hardik Garg    Guide: Dr. Swati Sharma , Galgotias University

---

## What this does

We combine an Xception backbone with spatial and channel attention to detect deepfake face videos at frame level. The attention module helps the model focus on regions like eye boundaries, mouth edges and hairlines — areas that face-swap methods tend to mess up.

We tested against MesoNet, VGGNet and standalone Xception on the DFD dataset:

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| MesoNet | 88.2% | 87.5% | 0.891 |
| VGGNet | 91.5% | 90.6% | 0.921 |
| Xception | 93.1% | 93.2% | 0.944 |
| **Ours** | **95.5%** | **95.4%** | **0.971** |

Ablation (same setup, just swapping attention components):

| Variant | Accuracy | F1 |
|---|---|---|
| Xception only | 91.8% | 91.2% |
| + channel attention | 93.4% | 92.9% |
| + spatial attention | 94.1% | 93.8% |
| + both (final model) | 95.5% | 95.4% |

---

## Files

```
model.py              — model architecture
preprocess.py         — frame extraction, MTCNN face detection, train/val/test split
train.py              — training
evaluate.py           — metrics + confusion matrix + ROC + grad-cam
inference.py          — run on a single image or video
create_sample_inputs.py  — makes test images if you don't have the dataset
demo.py               — quick demo without needing to train anything
kaggle_notebook.py    — everything in one file for running on Kaggle
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Quick demo (no dataset or trained model needed):
```bash
python create_sample_inputs.py
python demo.py
```

---

## Dataset

We used the [DFD dataset on Kaggle](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original)-dataset/data. Download it and put the videos like this:

```
data/raw/
    real/    <- original videos
    fake/    <- deepfake videos
```

---

## Running the full pipeline

```bash
# 1. extract faces from videos (takes a while)
python preprocess.py --raw data/raw --out data/processed

# 2. train — runs for up to 20 epochs with early stopping
python train.py --data data/processed

# 3. evaluate on test set
python evaluate.py --model saved_models/best_model.h5 --data data/processed

# 4. predict on a new image or video
python inference.py --input yourfile.jpg
python inference.py --input yourfile.mp4
```

If you want the grad-cam heatmap on an image:
```bash
python inference.py --input yourfile.jpg --heatmap
```

---

## Training setup

- TensorFlow 2.13, single NVIDIA T4 (Kaggle)  
- Adam, lr=1e-4, batch size 32, max 20 epochs  
- LR halved after 3 epochs of no improvement on val loss  
- Early stopping with patience=5  
- In practice it converged around epoch 14-18

## Running on Kaggle

Use `kaggle_notebook.py` — add the DFD dataset to your notebook from Settings → Add Data, then run the cells in order.

---

## Contact

Chetan Singh — chetansingh1274@gmail.com  
Hardik Garg — garghardik743@gmail.com  
Dr. Swati Sharma — swatisharma@galgotiasuniversity.edu.in
