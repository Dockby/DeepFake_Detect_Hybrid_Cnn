# quick demo — no training or dataset needed
# builds the model (downloads xception imagenet weights automatically)
# then predicts on whatever is in sample_inputs/
#
# if you have a trained model pass it with --model saved_models/best_model.h5

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import build_model

os.makedirs('results', exist_ok=True)
SAMPLE_DIR = 'sample_inputs'


def load_model(path):
    if path and os.path.exists(path):
        print(f'loading trained model: {path}')
        return tf.keras.models.load_model(path), True

    print('no trained model found — using imagenet weights (architecture demo only)')
    print('predictions will not be meaningful until you run train.py\n')
    m = build_model(freeze_backbone=True)
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m, False


def run(model, trained):
    files = sorted([f for f in os.listdir(SAMPLE_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not files:
        print(f'no images found in {SAMPLE_DIR}/')
        print('run: python create_sample_inputs.py')
        sys.exit(1)

    results = []
    for fname in files:
        path = os.path.join(SAMPLE_DIR, fname)
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0

        prob  = float(model.predict(np.expand_dims(img, 0), verbose=0)[0][0])
        label = 'FAKE' if prob >= 0.5 else 'REAL'
        conf  = prob if label == 'FAKE' else 1.0 - prob

        gt      = 'REAL' if 'real' in fname else ('FAKE' if 'fake' in fname else None)
        correct = (gt == label) if gt else None

        mark = ('✓' if correct else '✗') if correct is not None else '?'
        print(f'  {fname:<35} {label}  {conf:.1%}  {mark}')
        results.append((fname, img, label, conf, gt, correct))

    save_grid(results, trained)

    labelled = [(l, c) for _, _, l, _, gt, c in results if c is not None]
    if labelled:
        n_ok = sum(c for _, c in labelled)
        print(f'\n{n_ok}/{len(labelled)} correct on labelled samples')

    if not trained:
        print('\nnote: model was not trained on deepfake data — these are random predictions')

    print('grid saved -> results/demo_output.png')


def save_grid(results, trained):
    n    = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (fname, img, label, conf, gt, correct) in enumerate(results):
        ax    = axes[i]
        color = '#c0392b' if label == 'FAKE' else '#27ae60'

        ax.imshow(img)
        title = f'{label}  {conf:.1%}'
        if correct is not None:
            title += '  ' + ('✓' if correct else '✗')
        ax.set_title(title, color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3); sp.set_visible(True)

    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    status = 'trained model' if trained else 'untrained (imagenet weights only)'
    fig.suptitle(f'DeepFake Detector — {status}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/demo_output.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='saved_models/best_model.h5')
    args = ap.parse_args()

    model, trained = load_model(args.model)
    run(model, trained)
