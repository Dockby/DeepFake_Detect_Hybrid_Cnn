# ---------------------------------------------------------------
# Full training pipeline for Kaggle
# Dataset: DFD Entire Original Dataset (Tiwarekar)
# https://www.kaggle.com/datasets/soroush365/
#          deep-fake-detection-dfd-entire-original-dataset
#
# Add the dataset via: Settings -> Add Data -> search "DFD"
# Then run cells in order (or run this as a .py script)
# ---------------------------------------------------------------


# cell 1 — imports
import os, cv2, random, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from mtcnn import MTCNN
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

print('tf version:', tf.__version__)
print('gpu:', tf.config.list_physical_devices('GPU'))


# cell 2 — config
KAGGLE_REAL  = '/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/original_sequences/youtube/raw/videos'
KAGGLE_FAKE  = '/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/manipulated_sequences'

PROC   = '/kaggle/working/processed'
SAVE   = '/kaggle/working/models'
LOGS   = '/kaggle/working/logs'
PLOTS  = '/kaggle/working/results'
for d in [PROC, SAVE, LOGS, PLOTS]:
    os.makedirs(d, exist_ok=True)

IMG   = 224
BS    = 32
EPCH  = 20
LR    = 1e-4
CONF  = 0.95
SEED  = 42
np.random.seed(SEED); tf.random.set_seed(SEED)


# cell 3 — preprocessing
def preprocess_videos(vid_dir, label, out_root, detector,
                      splits=(0.75, 0.08, 0.17)):
    vids = [f for f in os.listdir(vid_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    random.shuffle(vids)

    n = len(vids)
    assigned = (
        [('train', v) for v in vids[:int(n*splits[0])]] +
        [('val',   v) for v in vids[int(n*splits[0]): int(n*(splits[0]+splits[1]))]] +
        [('test',  v) for v in vids[int(n*(splits[0]+splits[1])):]]
    )

    for split, vid in assigned:
        cap   = cv2.VideoCapture(os.path.join(vid_dir, vid))
        vfps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step  = max(1, int(vfps / 10))
        stem  = Path(vid).stem
        tmp   = f'/tmp/_f_{stem}'
        os.makedirs(tmp, exist_ok=True)

        i = j = 0
        while cap.isOpened():
            ok, fr = cap.read()
            if not ok: break
            if i % step == 0:
                cv2.imwrite(f'{tmp}/{j:06d}.jpg', fr)
                j += 1
            i += 1
        cap.release()

        dst = os.path.join(out_root, split, label)
        os.makedirs(dst, exist_ok=True)
        saved = 0

        for ff in sorted(os.listdir(tmp)):
            img = cv2.imread(os.path.join(tmp, ff))
            if img is None: continue
            det = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not det: continue
            b = max(det, key=lambda d: d['confidence'])
            if b['confidence'] < CONF: continue
            x, y, w, h = b['box']
            x, y = max(0,x), max(0,y)
            m = int(w * 0.2)
            face = cv2.resize(img[max(0,y-m):y+h+m, max(0,x-m):x+w+m], (IMG, IMG))
            cv2.imwrite(os.path.join(dst, f'{stem}_{ff}'), face)
            saved += 1

        shutil.rmtree(tmp, ignore_errors=True)
        print(f'[{label}] {stem}: {saved} faces -> {split}')


det = MTCNN()
preprocess_videos(KAGGLE_REAL, 'real', PROC, det)
for sub in os.listdir(KAGGLE_FAKE):
    fdir = os.path.join(KAGGLE_FAKE, sub, 'raw', 'videos')
    if os.path.isdir(fdir):
        preprocess_videos(fdir, 'fake', PROC, det)


# cell 4 — data generators
aug = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                          rotation_range=15, zoom_range=0.1,
                          width_shift_range=0.1, height_shift_range=0.1,
                          brightness_range=[0.8, 1.2])
ev  = ImageDataGenerator(rescale=1./255)
kw  = dict(target_size=(IMG,IMG), batch_size=BS, class_mode='binary', seed=SEED)

tr_gen  = aug.flow_from_directory(f'{PROC}/train', shuffle=True,  **kw)
val_gen = ev.flow_from_directory( f'{PROC}/val',   shuffle=False, **kw)
te_gen  = ev.flow_from_directory( f'{PROC}/test',  shuffle=False, **kw)
print(f'train:{tr_gen.samples} val:{val_gen.samples} test:{te_gen.samples}')


# cell 5 — model
def ch_att(x, r=8):
    ch = x.shape[-1]
    f1 = layers.Dense(ch//r, activation='relu', use_bias=False)
    f2 = layers.Dense(ch, use_bias=False)
    avg = layers.Reshape((1,1,ch))(layers.GlobalAveragePooling2D()(x))
    mx  = layers.Reshape((1,1,ch))(layers.GlobalMaxPooling2D()(x))
    return layers.Multiply()([x, layers.Activation('sigmoid')(f2(f1(avg))+f2(f1(mx)))])

def sp_att(x, k=7):
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx  = tf.reduce_max(x,  axis=-1, keepdims=True)
    m   = layers.Conv2D(1, k, padding='same', activation='sigmoid', use_bias=False)(
              layers.Concatenate(axis=-1)([avg, mx]))
    return layers.Multiply()([x, m])

backbone = Xception(include_top=False, weights='imagenet',
                    input_shape=(IMG, IMG, 3))
backbone.trainable = True
x = ch_att(backbone.output)
x = sp_att(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
model = Model(backbone.input, layers.Dense(1, activation='sigmoid')(x))

model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])


# cell 6 — train
cbs = [
    ModelCheckpoint(f'{SAVE}/best.h5', monitor='val_auc',
                    mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
    CSVLogger(f'{LOGS}/log.csv')
]

history = model.fit(tr_gen, validation_data=val_gen,
                    epochs=EPCH, callbacks=cbs)
model.save(f'{SAVE}/final.h5')


# cell 7 — accuracy/loss plots
ep = range(1, len(history.history['accuracy']) + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ep, history.history['accuracy'],     'o-', label='train')
ax[0].plot(ep, history.history['val_accuracy'], 's-', label='val')
ax[0].set_title('Accuracy'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, history.history['loss'],     'o-', label='train')
ax[1].plot(ep, history.history['val_loss'], 's-', label='val')
ax[1].set_title('Loss'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{PLOTS}/curves.png', dpi=150); plt.show()


# cell 8 — evaluate
best = tf.keras.models.load_model(f'{SAVE}/best.h5')
y_prob = best.predict(te_gen).ravel()
y_pred = (y_prob >= 0.5).astype(int)
y_true = te_gen.classes

print(classification_report(y_true, y_pred,
                             target_names=['Real', 'Fake'], digits=4))
print(f'AUC: {roc_auc_score(y_true, y_prob):.4f}')


# cell 9 — confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real','Fake'], yticklabels=['Real','Fake'],
            annot_kws={'size':18,'weight':'bold'})
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig(f'{PLOTS}/cm.png', dpi=150); plt.show()


# cell 10 — ROC
auc = roc_auc_score(y_true, y_prob)
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f'AUC={auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.fill_between(fpr, tpr, alpha=0.08)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{PLOTS}/roc.png', dpi=150); plt.show()
