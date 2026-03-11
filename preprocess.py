import os
import cv2
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE    = 224
FPS         = 10
CONF_THRESH = 0.95
MARGIN      = 0.20
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)


def extract_frames(video_path, out_dir, fps=FPS):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(native_fps / fps))

    idx = count = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f'{count:06d}.jpg'), frame)
            count += 1
        idx += 1
    cap.release()
    return count


def crop_face(img_path, detector, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    detections = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not detections:
        return False

    best = max(detections, key=lambda d: d['confidence'])
    if best['confidence'] < CONF_THRESH:
        return False

    x, y, w, h = best['box']
    x, y = max(0, x), max(0, y)
    mx, my = int(w * MARGIN), int(h * MARGIN)

    crop = img[max(0, y-my): y+h+my, max(0, x-mx): x+w+mx]
    cv2.imwrite(save_path, cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))
    return True


def run_preprocessing(raw_dir, out_dir):
    detector = MTCNN()

    # split ratios from the paper: 75/8/17
    splits = (0.75, 0.08, 0.17)

    for label in ('real', 'fake'):
        vid_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(vid_dir):
            print(f'skipping {vid_dir}, not found')
            continue

        videos = [f for f in os.listdir(vid_dir)
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        random.shuffle(videos)

        n = len(videos)
        n_tr = int(n * splits[0])
        n_v  = int(n * splits[1])

        # split at video level so no frames from the same video end up
        # in both train and test — important to avoid data leakage
        assigned = (
            [('train', v) for v in videos[:n_tr]] +
            [('val',   v) for v in videos[n_tr: n_tr + n_v]] +
            [('test',  v) for v in videos[n_tr + n_v:]]
        )

        for split, vid in assigned:
            stem    = Path(vid).stem
            tmp     = f'/tmp/_frames_{stem}'
            n_raw   = extract_frames(os.path.join(vid_dir, vid), tmp)

            dest = os.path.join(out_dir, split, label)
            os.makedirs(dest, exist_ok=True)

            saved = 0
            for f in sorted(os.listdir(tmp)):
                dst = os.path.join(dest, f'{stem}_{f}')
                if crop_face(os.path.join(tmp, f), detector, dst):
                    saved += 1

            shutil.rmtree(tmp, ignore_errors=True)
            print(f'[{label}] {stem}: {n_raw} frames -> {saved} faces -> {split}')

    print('done.')


def get_generators(data_dir, batch_size=32):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    common = dict(target_size=(IMG_SIZE, IMG_SIZE),
                  batch_size=batch_size,
                  class_mode='binary',
                  seed=SEED)

    tr = train_gen.flow_from_directory(os.path.join(data_dir, 'train'), shuffle=True,  **common)
    v  = val_gen.flow_from_directory(  os.path.join(data_dir, 'val'),   shuffle=False, **common)
    te = val_gen.flow_from_directory(  os.path.join(data_dir, 'test'),  shuffle=False, **common)

    return tr, v, te


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--out', default='data/processed')
    args = ap.parse_args()
    run_preprocessing(args.raw, args.out)
