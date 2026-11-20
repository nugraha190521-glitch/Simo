import os
os.environ['KMP_DUPLICATE_LIB_OK']   = 'TRUE'   # hindari bentrok OpenMP torch + tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'

import ast
import io
import threading

import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image, ImageOps

from predict import load_models, predict as run_predict

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'klasifikasi_model')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB

# ── Load semua model SEKALI saat server start ──────────────
MODELS = load_models(
    keras_path = os.path.join(MODEL_DIR, 'MobileNetV2-moluska-100.keras'),
    label_path = os.path.join(MODEL_DIR, 'moluska-8.txt'),
)

with open(os.path.join(MODEL_DIR, 'klasifikasi_metrics.txt'), 'r') as f:
    KL_METRICS = ast.literal_eval(f.read().strip())

_predict_lock = threading.Lock()   # satu inferensi pada satu waktu


# ── Halaman ────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('beranda'))


@app.route('/beranda')
def beranda():
    return render_template('beranda.html', daftar_kelas=list(MODELS['kl']['label_dict'].values()))


@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')


@app.route('/model-info')
def model_info():
    keras_model = MODELS['kl']['model']
    trainable   = sum(int(np.prod(w.shape)) for w in keras_model.trainable_weights)

    def rata2(kunci):   # macro average dari metrik per kelas, dalam persen
        nilai = [c[kunci] for c in KL_METRICS['classes'].values()]
        return round(sum(nilai) / len(nilai) * 100, 2)

    data_klasifikasi = {
        'nama_model'   : 'MobileNetV2',
        'framework'    : 'TensorFlow / Keras',
        'preprocessing': 'BiRefNet + crop persegi',
        'input_size'   : '224 × 224 px',
        'jumlah_kelas' : len(MODELS['kl']['label_dict']),
        'kelas'        : list(MODELS['kl']['label_dict'].values()),
        'total_params' : f"{keras_model.count_params():,}",
        'trainable'    : f"{trainable:,}",
        'accuracy_pct' : round(KL_METRICS['accuracy'] * 100, 2),
        'precision_pct': rata2('precision'),
        'recall_pct'   : rata2('recall'),
        'f1_pct'       : rata2('f1'),
        'classes'      : KL_METRICS['classes'],
        'best_epoch'   : KL_METRICS['best_epoch'],
    }
    return render_template('model_info.html', klasifikasi=data_klasifikasi)


# ── API Predict ────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File kosong'}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
        pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
    except Exception:
        return jsonify({'error': 'File bukan gambar yang valid'}), 400

    try:
        with _predict_lock:
            result = run_predict(pil_img, MODELS)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str('Terjadi kesalahan saat memproses gambar')}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'Ukuran file maksimal 10MB'}), 413


if __name__ == '__main__':
    # use_reloader=False: tanpa ini, model dimuat dua kali (BiRefNet + Keras) saat debug
    app.run(debug=False, use_reloader=False)
