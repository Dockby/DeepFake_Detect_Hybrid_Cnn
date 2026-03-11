import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score
)
from preprocess import get_generators

os.makedirs('results', exist_ok=True)


def evaluate(model_path, data_dir, batch_size=32):
    model = tf.keras.models.load_model(model_path)

    _, _, test_gen = get_generators(data_dir, batch_size)
    print(f'test samples: {test_gen.samples}')
    print(f'class map: {test_gen.class_indices}')

    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    acc  = float(np.mean(y_pred == y_true))
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print(f'\nAccuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1       : {f1:.4f}')
    print(f'AUC      : {auc:.4f}')
    print()
    print(classification_report(y_true, y_pred,
                                target_names=['Real', 'Fake'], digits=4))

    with open('results/metrics.txt', 'w') as f:
        f.write(f'Accuracy : {acc:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall   : {rec:.4f}\n')
        f.write(f'F1       : {f1:.4f}\n')
        f.write(f'AUC      : {auc:.4f}\n\n')
        f.write(classification_report(y_true, y_pred,
                                      target_names=['Real', 'Fake'], digits=4))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                annot_kws={'size': 18, 'weight': 'bold'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.fill_between(fpr, tpr, alpha=0.08)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=150)
    plt.close()

    print('results saved to results/')
    return auc


def plot_curves(log_path='logs/training_log.csv'):
    import pandas as pd
    if not os.path.exists(log_path):
        print(f'{log_path} not found, skipping')
        return

    df = pd.read_csv(log_path)
    ep = df['epoch'] + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ep, df['accuracy'],     'o-', label='train')
    axes[0].plot(ep, df['val_accuracy'], 's-', label='val')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim([0.6, 1.0])
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, df['loss'],     'o-', label='train')
    axes[1].plot(ep, df['val_loss'], 's-', label='val')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.close()
    print('curves saved')


def gradcam(model_path, img_path, out_path=None):
    import cv2
    model    = tf.keras.models.load_model(model_path)
    out_path = out_path or 'results/heatmap.png'

    try:
        conv_layer = model.get_layer('block14_sepconv2_act')
    except ValueError:
        conv_layer = [l for l in model.layers
                      if isinstance(l, tf.keras.layers.Conv2D)][-1]

    grad_model = tf.keras.Model(model.inputs,
                                [conv_layer.output, model.output])

    img = cv2.imread(img_path)
    inp = np.expand_dims(
        cv2.cvtColor(cv2.resize(img, (224, 224)),
                     cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
        axis=0
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(inp)
        score = pred[:, 0]

    grads   = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    heatmap = (conv_out[0].numpy() @ weights).clip(min=0)
    heatmap /= (heatmap.max() + 1e-8)

    h = cv2.resize(heatmap, (224, 224))
    colored  = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    original = cv2.resize(img, (224, 224))
    overlay  = cv2.addWeighted(original, 0.6, colored, 0.4, 0)

    cv2.imwrite(out_path, np.hstack([original, overlay]))

    label = 'FAKE' if pred[0][0] >= 0.5 else 'REAL'
    conf  = float(pred[0][0]) if label == 'FAKE' else 1 - float(pred[0][0])
    print(f'{label} ({conf:.1%}) — heatmap saved to {out_path}')
    return label, conf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',   default='saved_models/best_model.h5')
    ap.add_argument('--data',    default='data/processed')
    ap.add_argument('--log',     default='logs/training_log.csv')
    ap.add_argument('--heatmap', default=None)
    args = ap.parse_args()

    evaluate(args.model, args.data)
    plot_curves(args.log)
    if args.heatmap:
        gradcam(args.model, args.heatmap)
