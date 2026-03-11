

import os
import cv2
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hyper-params (match paper)
IMG_SIZE     = 224
FPS_EXTRACT  = 10       # frames per second to extract
MTCNN_THRESH = 0.95     # discard faces below this confidence
FACE_MARGIN  = 0.20     # extra margin around detected face box
SPLIT_RATIOS = (0.75, 0.08, 0.17)   # train / val / test
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



#  Step 1: Extract frames from video


def extract_frames(video_path: str, tmp_dir: str, fps: int = FPS_EXTRACT) -> int:
    """
    Save one frame every (video_fps / fps) frames as JPEG.
    Returns number of frames saved.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step   = max(1, int(video_fps / fps))

    frame_idx = saved = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            cv2.imwrite(os.path.join(tmp_dir, f'{saved:06d}.jpg'), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved



#  Step 2: Detect & crop face


def crop_face(image_path: str, detector: MTCNN, out_path: str) -> bool:
    """
    Detect best face, apply margin, resize to IMG_SIZE.
    Returns True if saved successfully.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces   = detector.detect_faces(rgb)
    if not faces:
        return False

    best = max(faces, key=lambda f: f['confidence'])
    if best['confidence'] < MTCNN_THRESH:
        return False

    x, y, w, h = best['box']
    x, y = max(0, x), max(0, y)
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x1 = max(0, x - mx);       y1 = max(0, y - my)
    x2 = min(img.shape[1], x + w + mx)
    y2 = min(img.shape[0], y + h + my)

    face = cv2.resize(img[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(out_path, face)
    return True



#  Main pipeline


def run_preprocessing(raw_dir: str, out_dir: str):
    """
    Full preprocessing for real + fake video folders.
    Split is done at VIDEO level to prevent leakage.
    """
    detector = MTCNN()

    for label in ('real', 'fake'):
        video_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(video_dir):
            print(f'[WARN] {video_dir} not found, skipping.')
            continue

        videos = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])
        random.shuffle(videos)

        n_train = int(len(videos) * SPLIT_RATIOS[0])
        n_val   = int(len(videos) * SPLIT_RATIOS[1])
        split_map = (
            [('train', v) for v in videos[:n_train]] +
            [('val',   v) for v in videos[n_train:n_train + n_val]] +
            [('test',  v) for v in videos[n_train + n_val:]]
        )

        total_saved = {'train': 0, 'val': 0, 'test': 0}

        for split, vid_name in split_map:
            vid_path = os.path.join(video_dir, vid_name)
            vid_stem = Path(vid_name).stem
            tmp_dir  = f'/tmp/_df_frames_{vid_stem}'

            # 1. Extract raw frames
            n_raw = extract_frames(vid_path, tmp_dir)

            # 2. Crop faces and save
            dest_dir = os.path.join(out_dir, split, label)
            os.makedirs(dest_dir, exist_ok=True)

            n_saved = 0
            for frame_file in sorted(os.listdir(tmp_dir)):
                src = os.path.join(tmp_dir, frame_file)
                dst = os.path.join(dest_dir, f'{vid_stem}_{frame_file}')
                if crop_face(src, detector, dst):
                    n_saved += 1

            shutil.rmtree(tmp_dir, ignore_errors=True)
            total_saved[split] += n_saved
            print(f'  [{label}] {vid_stem}: {n_raw} frames → {n_saved} faces → {split}')

        print(f'\n[{label.upper()}] Totals: '
              f'train={total_saved["train"]} | '
              f'val={total_saved["val"]} | '
              f'test={total_saved["test"]}\n')

    print('✅  Preprocessing complete.')


#  Data generators for training


def get_generators(processed_dir: str, batch_size: int = 32):
    """
    Returns (train_gen, val_gen, test_gen).
    Augmentation applied ONLY during training (matches paper).
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.80, 1.20]
    )
    eval_aug = ImageDataGenerator(rescale=1.0 / 255)

    def _flow(aug, split, shuffle):
        return aug.flow_from_directory(
            os.path.join(processed_dir, split),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=shuffle,
            seed=RANDOM_SEED
        )

    train_gen = _flow(train_aug, 'train', shuffle=True)
    val_gen   = _flow(eval_aug,  'val',   shuffle=False)
    test_gen  = _flow(eval_aug,  'test',  shuffle=False)

    return train_gen, val_gen, test_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw',
                        help='Root dir with real/ and fake/ video folders')
    parser.add_argument('--out', default='data/processed',
                        help='Output dir for face-crop images')
    args = parser.parse_args()
    run_preprocessing(args.raw, args.out)
