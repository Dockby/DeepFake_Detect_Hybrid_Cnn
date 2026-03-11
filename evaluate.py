"""
evaluate.py
===========
Full evaluation on the DFD test set.

Generates:
  results/metrics.txt
  results/confusion_matrix.png
  results/roc_curve.png
  results/training_curves.png

Usage:
    python evaluate.py
    python evaluate.py --model saved_models/best_model.h5 --data data/processed
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')           # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
from preprocess import get_generators

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


#  Main evaluation


def evaluate(model_path: str, data_dir: str, batch_size: int = 32):
    """Run inference on held-out test set and report all metrics."""

    print(f'Loading model: {model_path}')
    model = tf.keras.models.load_model(model_path)

    _, _, test_gen = get_generators(data_dir, batch_size)
    print(f'Test samples : {test_gen.samples}')
    print(f'Class map    : {test_gen.class_indices}')   # {fake:0, real:1} or similar

    #  Predict
    print('Running predictions...')
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    # Metrics 
    auc  = roc_auc_score(y_true, y_prob)
    acc  = float(np.mean(y_pred == y_true))
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=['Real', 'Fake'],
        digits=4
    )

    summary = (
        '\n' + '='*55 + '\n'
        'TEST SET RESULTS (Held-out, never seen during training)\n'
        + '='*55 + '\n'
        f'Accuracy  : {acc:.4f}  ({acc*100:.1f}%)\n'
        f'Precision : {prec:.4f}  ({prec*100:.1f}%)\n'
        f'Recall    : {rec:.4f}  ({rec*100:.1f}%)\n'
        f'F1-Score  : {f1:.4f}  ({f1*100:.1f}%)\n'
        f'AUC       : {auc:.4f}\n'
        + '-'*55 + '\n'
        + report
    )
    print(summary)

    # Save text
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(summary)
    print(f'Metrics saved → {metrics_path}')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        annot_kws={'size': 18, 'weight': 'bold'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label',      fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, pad=10)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f'Confusion matrix → {cm_path}')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, color='steelblue',
             label=f'Hybrid CNN-Attention (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate',  fontsize=12)
    plt.title('ROC Curve',            fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f'ROC curve → {roc_path}')

    return {'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc}


 
#  Training curves


def plot_training_curves(log_csv: str = 'logs/training_log.csv'):
    """Plot accuracy & loss from training CSV log."""
    import pandas as pd

    if not os.path.exists(log_csv):
        print(f'[WARN] Log not found: {log_csv}')
        return

    df  = pd.read_csv(log_csv)
    ep  = df['epoch'] + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy
    axes[0].plot(ep, df['accuracy'],     'o-', label='Training',   color='steelblue')
    axes[0].plot(ep, df['val_accuracy'], 's-', label='Validation', color='tomato')
    axes[0].set_title('Training and Validation Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0.6, 1.0])

    # Loss
    axes[1].plot(ep, df['loss'],     'o-', label='Training',   color='steelblue')
    axes[1].plot(ep, df['val_loss'], 's-', label='Validation', color='tomato')
    axes[1].set_title('Training and Validation Loss', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f'Training curves → {curve_path}')


# 
#  Grad-CAM attention heatmap
# 

def generate_gradcam(model_path: str, image_path: str, out_path: str = None):
    """
    Generate Grad-CAM heatmap for a single image.
    Saves side-by-side: original | heatmap overlay.
    """
    import cv2

    model   = tf.keras.models.load_model(model_path)
    out_path = out_path or os.path.join(RESULTS_DIR, 'attention_heatmap.png')

    # Build grad model using last Xception conv layer
    try:
        last_conv = model.get_layer('block14_sepconv2_act')
    except ValueError:
        # fallback: use last Conv2D
        last_conv = [l for l in model.layers
                     if isinstance(l, tf.keras.layers.Conv2D)][-1]

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output]
    )

    # Load & preprocess
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(
        cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB
    )
    inp = np.expand_dims(img_rgb.astype(np.float32) / 255.0, 0)

    # Gradient tape
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(inp)
        score = pred[:, 0]

    grads       = tape.gradient(score, conv_out)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    heatmap     = (conv_out[0].numpy() @ pooled).clip(min=0)
    heatmap     = heatmap / (heatmap.max() + 1e-8)

    # Overlay
    heat_resized = cv2.resize(heatmap, (224, 224))
    heat_colored = cv2.applyColorMap(
        (heat_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(
        cv2.resize(img_bgr, (224, 224)), 0.6,
        heat_colored, 0.4, 0
    )

    # Side-by-side
    side_by_side = np.hstack([
        cv2.resize(img_bgr, (224, 224)),
        overlay
    ])
    cv2.imwrite(out_path, side_by_side)

    label = 'FAKE' if pred[0][0] >= 0.5 else 'REAL'
    conf  = float(pred[0][0]) if label == 'FAKE' else 1 - float(pred[0][0])
    print(f'Prediction : {label}  (confidence: {conf:.1%})')
    print(f'Heatmap    → {out_path}')
    return label, conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default='saved_models/best_model.h5')
    parser.add_argument('--data',     default='data/processed')
    parser.add_argument('--log',      default='logs/training_log.csv')
    parser.add_argument('--heatmap',  default=None,
                        help='Path to single image for Grad-CAM heatmap')
    args = parser.parse_args()

    evaluate(args.model, args.data)
    plot_training_curves(args.log)
    if args.heatmap:
        generate_gradcam(args.model, args.heatmap)
