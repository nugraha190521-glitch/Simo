"""
utils/preprocessing.py
Tanggung jawab:
  - Hapus background dengan BiRefNet (ZhengPeng7/BiRefNet)
  - Tempel objek di latar putih, crop ke objek, jadikan persegi 1:1

Logika sama dengan hapusBg.py yang dipakai untuk menyiapkan dataset
training, supaya gambar saat inferensi mirip dengan gambar saat training.
"""

import os
import numpy as np
from PIL import Image

MODEL_ID       = os.environ.get('BIREFNET_MODEL', 'ZhengPeng7/BiRefNet')  # bisa diisi path folder lokal
INPUT_SIZE     = 1024   # ukuran input BiRefNet
MARGIN         = 0.05   # margin putih di sekeliling objek (5% sisi terpanjang)
MASK_THRESHOLD = 20     # piksel mask > nilai ini dianggap bagian objek (0-255)


# ── Load ──────────────────────────────────────────────────

def load_birefnet(model_id: str = MODEL_ID) -> dict:
    """
    Load BiRefNet sekali saat server start.
    Pertama kali dijalankan, bobot (~1 GB) diunduh dari Hugging Face lalu di-cache.
    """
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True
    ).to(device).float().eval()

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return {'model': model, 'transform': transform, 'device': device}


# ── Inferensi BiRefNet ────────────────────────────────────

def _predict_mask(img: Image.Image, bg: dict) -> Image.Image:
    """Hasilkan mask objek (mode 'L', ukuran sama dengan gambar asli)."""
    import torch
    from torchvision import transforms

    x = bg['transform'](img).unsqueeze(0).to(bg['device'])
    with torch.no_grad():
        pred = bg['model'](x)[-1].sigmoid().cpu()[0].squeeze()
    return transforms.ToPILImage()(pred).resize(img.size, Image.LANCZOS)


# ── Komposit + crop (tanpa torch, mudah diuji) ────────────

def composite_and_crop(img: Image.Image, mask: Image.Image,
                       margin: float = MARGIN,
                       threshold: int = MASK_THRESHOLD) -> Image.Image:
    """
    Tempel objek ke latar putih, crop ke kotak pembatas objek,
    lalu taruh di tengah kanvas putih persegi.
    """
    putih = Image.new('RGB', img.size, (255, 255, 255))
    hasil = Image.composite(img, putih, mask)

    ys, xs = np.where(np.array(mask) > threshold)
    if len(xs) == 0:   # objek tidak terdeteksi -> pakai gambar penuh
        x1, y1, x2, y2 = 0, 0, img.width - 1, img.height - 1
    else:
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()

    hasil = hasil.crop((int(x1), int(y1), int(x2) + 1, int(y2) + 1))
    w, h  = hasil.size

    pad    = int(max(w, h) * margin)
    sisi   = max(w, h) + 2 * pad
    kanvas = Image.new('RGB', (sisi, sisi), (255, 255, 255))
    kanvas.paste(hasil, ((sisi - w) // 2, (sisi - h) // 2))
    return kanvas


def remove_background(pil_img: Image.Image, bg: dict) -> Image.Image:
    """
    Parameters:
        pil_img : PIL.Image RGB (sudah tegak / exif_transpose)
        bg      : dict hasil load_birefnet()

    Returns:
        PIL.Image RGB persegi, background putih, objek di tengah
    """
    mask = _predict_mask(pil_img, bg)
    return composite_and_crop(pil_img, mask)
