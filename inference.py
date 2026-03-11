

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

IMG_SIZE    = 224
MTCNN_CONF  = 0.95
MODEL_PATH  = 'saved_models/best_model.h5'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


#  Helpers


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize and normalise a face crop for the model."""
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    return np.expand_dims(face, axis=0)


def detect_face(image_bgr: np.ndarray, detector: MTCNN):
    """
    Return the largest detected face crop (with margin)
    or None if no confident detection.
    """
    rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        return None

    best = max(faces, key=lambda f: f['confidence'])
    if best['confidence'] < MTCNN_CONF:
        return None

    x, y, w, h = best['box']
    x, y = max(0, x), max(0, y)
    margin = 0.20
    mx = int(w * margin);  my = int(h * margin)
    x1 = max(0, x - mx);   y1 = max(0, y - my)
    x2 = min(image_bgr.shape[1], x + w + mx)
    y2 = min(image_bgr.shape[0], y + h + my)
    return image_bgr[y1:y2, x1:x2]



#  Single image


def predict_image(model, detector, image_path: str,
                  save_heatmap: bool = False) -> dict:
    """
    Predict real/fake for one image.
    Returns dict with keys: label, confidence, raw_prob.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Cannot read image: {image_path}'}

    face = detect_face(img, detector)
    if face is None:
        # Fallback: use full image (resize to 224×224)
        print('[WARN] No face detected — using full image.')
        face = img

    inp  = preprocess_face(face)
    prob = float(model.predict(inp, verbose=0)[0][0])

    label = 'FAKE' if prob >= 0.5 else 'REAL'
    conf  = prob if label == 'FAKE' else 1.0 - prob

    result = {'label': label, 'confidence': conf, 'raw_prob': prob}

    if save_heatmap:
        _save_heatmap(model, inp, face, image_path)

    return result



#  Video  (frame-by-frame, majority vote)


def predict_video(model, detector, video_path: str,
                  fps: int = 5) -> dict:
    """
    Sample frames at `fps` rate, predict each face,
    return majority-vote decision.
    """
    cap        = cv2.VideoCapture(video_path)
    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step       = max(1, int(video_fps / fps))

    probs      = []
    frame_idx  = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            face = detect_face(frame, detector) or frame
            inp  = preprocess_face(face)
            p    = float(model.predict(inp, verbose=0)[0][0])
            probs.append(p)
        frame_idx += 1
    cap.release()

    if not probs:
        return {'error': 'No frames could be analysed.'}

    avg_prob = float(np.mean(probs))
    label    = 'FAKE' if avg_prob >= 0.5 else 'REAL'
    conf     = avg_prob if label == 'FAKE' else 1.0 - avg_prob

    return {
        'label':           label,
        'confidence':      conf,
        'frames_analysed': len(probs),
        'avg_fake_prob':   avg_prob
    }



#  Optional Grad-CAM heatmap


def _save_heatmap(model, inp: np.ndarray,
                  face_bgr: np.ndarray, source_path: str):
    try:
        last_conv = [l for l in model.layers
                     if 'sepconv2_act' in l.name or
                     isinstance(l, tf.keras.layers.Conv2D)][-1]
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(inp)
            score = pred[:, 0]
        grads   = tape.gradient(score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        heatmap = (conv_out[0].numpy() @ pooled).clip(min=0)
        heatmap /= (heatmap.max() + 1e-8)
        heat_r  = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        colored = cv2.applyColorMap((heat_r * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
        face_r  = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        overlay = cv2.addWeighted(face_r, 0.6, colored, 0.4, 0)
        out_p   = os.path.join(
            RESULTS_DIR,
            os.path.basename(source_path).rsplit('.', 1)[0] + '_heatmap.png'
        )
        cv2.imwrite(out_p, np.hstack([face_r, overlay]))
        print(f'Heatmap saved → {out_p}')
    except Exception as e:
        print(f'[WARN] Could not generate heatmap: {e}')



#  CLI


def main():
    parser = argparse.ArgumentParser(
        description='DeepFake Inference — Hybrid CNN-Attention Model'
    )
    parser.add_argument('--input',    required=True,
                        help='Path to image (.jpg/.png) or video (.mp4/.avi)')
    parser.add_argument('--model',    default=MODEL_PATH,
                        help='Path to trained .h5 model file')
    parser.add_argument('--heatmap',  action='store_true',
                        help='Save Grad-CAM heatmap (images only)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'[ERROR] Model not found: {args.model}')
        print('        Run train.py first, or use demo.py for a quick test.')
        return

    print(f'Loading model   : {args.model}')
    model    = tf.keras.models.load_model(args.model)
    detector = MTCNN()

    ext      = os.path.splitext(args.input)[1].lower()
    is_video = ext in ('.mp4', '.avi', '.mov', '.mkv')

    print(f'Analysing       : {args.input}')
    if is_video:
        result = predict_video(model, detector, args.input)
    else:
        result = predict_image(model, detector, args.input,
                               save_heatmap=args.heatmap)

    #  Print result
    print('\n' + '='*45)
    if 'error' in result:
        print(f'  ERROR : {result["error"]}')
    else:
        emoji  = '🚨' if result['label'] == 'FAKE' else '✅'
        print(f'  {emoji}  Prediction  : {result["label"]}')
        print(f'      Confidence  : {result["confidence"]:.1%}')
        if is_video:
            print(f'      Frames used : {result["frames_analysed"]}')
    print('='*45 + '\n')


if __name__ == '__main__':
    main()
