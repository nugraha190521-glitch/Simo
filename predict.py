"""
predict.py — Pipeline utama inferensi

  1. preprocessing.py → BiRefNet: hapus background + crop persegi
  2. klasifikasi.py   → klasifikasi genus moluska (MobileNetV2)
  3. similarity       → gatekeeper: tolak gambar yang bukan moluska
"""

import base64
import io

from PIL import Image

from utils.preprocessing import load_birefnet, remove_background
from utils.klasifikasi   import load_klasifikasi, predict_klasifikasi

LOW_CONFIDENCE = 0.60
MIN_SIMILARITY = 0.65


def load_models(keras_path: str, label_path: str) -> dict:
    """Load semua komponen sekali saat app.py start."""
    return {
        'bg': load_birefnet(),
        'kl': load_klasifikasi(keras_path, label_path),
    }


def _to_data_url(img: Image.Image, max_side: int = 512) -> str:
    """Hasil preprocessing -> data URL PNG kecil, untuk ditampilkan di UI."""
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def predict(pil_img: Image.Image, models: dict) -> dict:
    """
    Returns dict:
      - label, confidence, all_probs : hasil klasifikasi
      - similarity     : kemiripan fitur dengan centroid kelas terdekat
      - is_mollusc     : False bila similarity < MIN_SIMILARITY
      - low_confidence : True bila moluska terdeteksi tapi confidence < LOW_CONFIDENCE
      - processed_image: data URL gambar hasil BiRefNet
    """
    processed  = remove_background(pil_img, models['bg'])
    result     = predict_klasifikasi(processed, models['kl'])
    is_mollusc = result['similarity'] >= MIN_SIMILARITY

    return {
        'label'          : result['label'],
        'confidence'     : result['confidence'],
        'all_probs'      : result['all_probs'],
        'similarity'     : result['similarity'],
        'is_mollusc'     : is_mollusc,
        'low_confidence' : is_mollusc and result['confidence'] < LOW_CONFIDENCE,
        'processed_image': _to_data_url(processed),
    }