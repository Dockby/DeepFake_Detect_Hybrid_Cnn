import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from model import build_model
from preprocess import get_generators

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train(data_dir, batch_size=32, epochs=20, lr=1e-4,
          save_dir='saved_models', log_dir='logs'):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f'using GPU: {gpus[0].name}')
    else:
        print('no GPU found, running on CPU — this will be slow')

    tr, val, _ = get_generators(data_dir, batch_size)
    print(f'train: {tr.samples} | val: {val.samples}')
    print(f'classes: {tr.class_indices}')

    model = build_model(freeze_backbone=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    best_path  = os.path.join(save_dir, 'best_model.h5')
    callbacks = [
        ModelCheckpoint(best_path, monitor='val_auc', mode='max',
                        save_best_only=True, verbose=1),
        # stopped after val_loss didn't improve for 5 epochs consistently
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        # halve LR after 3 epochs plateau — helped with convergence
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(log_dir, 'training_log.csv'))
    ]

    history = model.fit(tr, validation_data=val,
                        epochs=epochs, callbacks=callbacks)

    model.save(os.path.join(save_dir, 'final_model.h5'))

    best_ep  = int(np.argmax(history.history['val_auc'])) + 1
    best_auc = max(history.history['val_auc'])
    print(f'\nbest epoch: {best_ep}  |  val AUC: {best_auc:.4f}')
    print(f'model saved to {save_dir}/')
    return history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',       default='data/processed')
    ap.add_argument('--batch_size', default=32,   type=int)
    ap.add_argument('--epochs',     default=20,   type=int)
    ap.add_argument('--lr',         default=1e-4, type=float)
    ap.add_argument('--save_dir',   default='saved_models')
    ap.add_argument('--log_dir',    default='logs')
    args = ap.parse_args()
    train(args.data, args.batch_size, args.epochs,
          args.lr, args.save_dir, args.log_dir)
