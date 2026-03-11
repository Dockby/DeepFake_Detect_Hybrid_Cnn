import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

IMG_SIZE   = 224
CONF_THRESH = 0.95
os.makedirs('results', exist_ok=True)


def get_face(img, detector):
    detections = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not detections:
        return None

    best = max(detections, key=lambda d: d['confidence'])
    if best['confidence'] < CONF_THRESH:
        return None

    x, y, w, h = best['box']
    x, y = max(0, x), max(0, y)
    m = int(w * 0.2)
    return img[max(0, y-m): y+h+m, max(0, x-m): x+w+m]


def preprocess(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return np.expand_dims(face, 0)


def predict_image(model, detector, path, save_heatmap=False):
    img = cv2.imread(path)
    if img is None:
        return {'error': f'could not read {path}'}

    face = get_face(img, detector)
    if face is None:
        print('no face detected, using full image')
        face = img

    prob  = float(model.predict(preprocess(face), verbose=0)[0][0])
    label = 'FAKE' if prob >= 0.5 else 'REAL'
    conf  = prob if label == 'FAKE' else 1.0 - prob

    if save_heatmap:
        _heatmap(model, preprocess(face), face, path)

    return {'label': label, 'confidence': conf, 'prob': prob}


def predict_video(model, detector, path, fps=5):
    cap   = cv2.VideoCapture(path)
    vfps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step  = max(1, int(vfps / fps))
    probs = []
    idx   = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            face = get_face(frame, detector) or frame
            probs.append(float(model.predict(preprocess(face), verbose=0)[0][0]))
        idx += 1
    cap.release()

    if not probs:
        return {'error': 'no frames processed'}

    avg   = float(np.mean(probs))
    label = 'FAKE' if avg >= 0.5 else 'REAL'
    conf  = avg if label == 'FAKE' else 1.0 - avg

    return {'label': label, 'confidence': conf,
            'frames': len(probs), 'avg_prob': avg}


def _heatmap(model, inp, face_crop, source):
    try:
        last = [l for l in model.layers
                if 'sepconv2_act' in l.name
                or isinstance(l, tf.keras.layers.Conv2D)][-1]
        gm = tf.keras.Model(model.inputs, [last.output, model.output])

        with tf.GradientTape() as tape:
            co, pred = gm(inp)
        grads = tape.gradient(pred[:, 0], co)
        w = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        h = (co[0].numpy() @ w).clip(min=0)
        h /= (h.max() + 1e-8)

        h_r = cv2.resize(h, (IMG_SIZE, IMG_SIZE))
        col = cv2.applyColorMap((h_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        fc  = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        ov  = cv2.addWeighted(fc, 0.6, col, 0.4, 0)

        out = os.path.join('results',
                           os.path.basename(source).rsplit('.', 1)[0] + '_heatmap.png')
        cv2.imwrite(out, np.hstack([fc, ov]))
        print(f'heatmap -> {out}')
    except Exception as e:
        print(f'heatmap failed: {e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',   required=True)
    ap.add_argument('--model',   default='saved_models/best_model.h5')
    ap.add_argument('--heatmap', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f'model not found at {args.model} — run train.py first')
        return

    model    = tf.keras.models.load_model(args.model)
    detector = MTCNN()

    ext      = os.path.splitext(args.input)[1].lower()
    is_video = ext in ('.mp4', '.avi', '.mov', '.mkv')

    result = (predict_video(model, detector, args.input)
              if is_video
              else predict_image(model, detector, args.input, args.heatmap))

    print('\n' + '-'*40)
    if 'error' in result:
        print(f'error: {result["error"]}')
    else:
        print(f'prediction : {result["label"]}')
        print(f'confidence : {result["confidence"]:.1%}')
        if is_video:
            print(f'frames used: {result["frames"]}')
    print('-'*40)


if __name__ == '__main__':
    main()
