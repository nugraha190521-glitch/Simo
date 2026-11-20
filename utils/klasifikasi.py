"""
utils/klasifikasi.py
Tanggung jawab:
  - Load model MobileNetV2 (.keras) dan label kelas dari .txt
  - Inferensi klasifikasi: kembalikan semua probabilitas kelas
"""

import ast
import tensorflow as tf
from PIL import Image
import numpy as np
import os

from utils.transforms import prepare_for_mobilenet


def load_klasifikasi(keras_path, label_path):
    model = tf.keras.models.load_model(keras_path)
    with open(label_path, 'r') as f:
        label_dict = ast.literal_eval(f.read().strip())

    feat = tf.keras.Sequential([model.layers[0], model.layers[1]])
    cent_path = os.path.join(os.path.dirname(keras_path), 'centroids.npy')
    centroids = np.load(cent_path)

    return {'model': model, 'label_dict': label_dict, 'feat': feat, 'centroids': centroids}



def predict_klasifikasi(pil_img, kl):
    arr   = prepare_for_mobilenet(pil_img)
    preds = kl['model'].predict(arr, verbose=0)[0]

    v   = kl['feat'].predict(arr, verbose=0)[0]
    v   = v / np.linalg.norm(v)
    sim = float((kl['centroids'] @ v).max())

    all_probs = [{'label': kl['label_dict'][i], 'prob': round(float(preds[i]), 6)}
                 for i in range(len(preds))]
    all_probs.sort(key=lambda x: x['prob'], reverse=True)

    return {
        
    'label': all_probs[0]['label'], 
    'confidence': all_probs[0]['prob'],
    'all_probs': all_probs, 
    'similarity': sim,
    }
