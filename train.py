"""
train.py
========
Train the Hybrid CNN-Attention deepfake detector.

Usage (after preprocessing):
    python train.py --data data/processed

Or with custom settings:
    python train.py --data data/processed --epochs 20 --batch_size 32 --lr 0.0001

Hardware: NVIDIA T4 GPU (Kaggle/Colab) — matches paper.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)

from model import build_model
from preprocess import get_generators

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Defaults (exact values from paper) ───────────────────────
DEFAULTS = dict(
    data       = 'data/processed',
    batch_size = 32,
    epochs     = 20,
    lr         = 1e-4,
    save_dir   = 'saved_models',
    log_dir    = 'logs',
)


def train(cfg: dict):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'],  exist_ok=True)

    # ── GPU config ────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f'GPU: {gpus[0].name}')
    else:
        print('No GPU found — training on CPU (will be slow).')

    # ── Data ──────────────────────────────────────────────────
    print('\nLoading data...')
    train_gen, val_gen, _ = get_generators(cfg['data'], cfg['batch_size'])
    print(f'  Train: {train_gen.samples} images  ({len(train_gen)} batches)')
    print(f'  Val  : {val_gen.samples}   images  ({len(val_gen)} batches)')
    print(f'  Classes: {train_gen.class_indices}')

    # ── Model ─────────────────────────────────────────────────
    print('\nBuilding model...')
    model = build_model(input_shape=(224, 224, 3), freeze_backbone=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # Print only trainable param count
    trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
    total     = model.count_params()
    print(f'  Total params     : {total:,}')
    print(f'  Trainable params : {trainable:,}')

    # ── Callbacks ─────────────────────────────────────────────
    ckpt_path = os.path.join(cfg['save_dir'], 'best_model.h5')
    callbacks = [
        ModelCheckpoint(
            filepath    = ckpt_path,
            monitor     = 'val_auc',
            mode        = 'max',
            save_best_only = True,
            verbose     = 1
        ),
        EarlyStopping(
            monitor             = 'val_loss',
            patience            = 5,            # paper: patience=5
            restore_best_weights = True,
            verbose             = 1
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,                     # paper: halve LR
            patience = 3,                       # paper: after 3 epochs
            min_lr   = 1e-7,
            verbose  = 1
        ),
        CSVLogger(os.path.join(cfg['log_dir'], 'training_log.csv'))
    ]

    # ── Train ─────────────────────────────────────────────────
    print(f'\nTraining for up to {cfg["epochs"]} epochs...')
    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = cfg['epochs'],
        callbacks       = callbacks,
        verbose         = 1
    )

    # Save final weights too
    final_path = os.path.join(cfg['save_dir'], 'final_model.h5')
    model.save(final_path)

    # ── Summary ───────────────────────────────────────────────
    best_epoch = int(np.argmax(history.history['val_auc'])) + 1
    best_auc   = max(history.history['val_auc'])
    best_acc   = history.history['val_accuracy'][best_epoch - 1]

    print('\n' + '='*50)
    print('TRAINING COMPLETE')
    print('='*50)
    print(f'  Best epoch : {best_epoch}')
    print(f'  Val AUC    : {best_auc:.4f}')
    print(f'  Val Acc    : {best_acc:.4f}')
    print(f'  Best model : {ckpt_path}')
    print(f'  Final model: {final_path}')
    print('='*50)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Hybrid CNN-Attention DeepFake Detector'
    )
    parser.add_argument('--data',       default=DEFAULTS['data'])
    parser.add_argument('--batch_size', default=DEFAULTS['batch_size'], type=int)
    parser.add_argument('--epochs',     default=DEFAULTS['epochs'],     type=int)
    parser.add_argument('--lr',         default=DEFAULTS['lr'],         type=float)
    parser.add_argument('--save_dir',   default=DEFAULTS['save_dir'])
    parser.add_argument('--log_dir',    default=DEFAULTS['log_dir'])
    args = parser.parse_args()
    train(vars(args))
