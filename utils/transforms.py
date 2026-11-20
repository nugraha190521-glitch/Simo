"""
utils/transforms.py
Tanggung jawab:
  - Resize gambar ke 224×224 (ukuran input MobileNetV2)
  - Normalisasi piksel persis seperti saat training

Saat training (klasifikasi.ipynb) dipakai:
    ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input)
yaitu piksel diskalakan ke rentang [-1, 1]  ->  x / 127.5 - 1
BUKAN x / 255. Jika tidak sama, akurasi saat inferensi akan turun.
"""

import numpy as np
from PIL import Image

INPUT_SIZE = (224, 224)

# flow_from_dataframe (Keras) me-resize dengan interpolasi 'nearest' secara
# default, jadi inferensi memakai yang sama agar distribusi piksel identik
# dengan data training. Ganti ke Image.BILINEAR bila ingin hasil lebih halus.
RESIZE_METHOD = Image.NEAREST


def prepare_for_mobilenet(pil_img: Image.Image) -> np.ndarray:
    """
    Resize ke 224×224 lalu skala piksel ke [-1, 1].

    Returns:
        numpy array shape (1, 224, 224, 3), dtype float32
    """
    img = pil_img.convert('RGB').resize(INPUT_SIZE, RESIZE_METHOD)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)
