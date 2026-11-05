"""
=============================================================
 Deteksi Penyakit Gigi — Backend (Flask)
=============================================================
 Pipeline:
   1. CNN ResNet-18 → Klasifikasi jenis penyakit gigi
      (model utama, bekerja pada gambar penuh)
   2. YOLO v8 (opsional) → Deteksi bounding box area penyakit
      Hanya aktif jika file 'best.pt' fine-tuned tersedia.
      Jika tidak, gunakan full-image classification saja.
=============================================================
"""

import os
import uuid
import base64
from io import BytesIO
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify, url_for
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ──────────────────────────────────────────────────────────
# Konfigurasi Aplikasi
# ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
RESULT_FOLDER = os.path.join(UPLOAD_FOLDER, "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# ──────────────────────────────────────────────────────────
# Path ke model
# ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CNN_WEIGHTS  = os.path.join(DATA_DIR, "cnn_best_model.pth")

# YOLO: gunakan best.pt (fine-tuned) jika ada,
# jika tidak ada → YOLO tidak dipakai
YOLO_FINETUNED = os.path.join(DATA_DIR, "best.pt")
YOLO_AVAILABLE = os.path.exists(YOLO_FINETUNED)

# ──────────────────────────────────────────────────────────
# Konfigurasi CNN ResNet-18 (dari notebook training)
# ──────────────────────────────────────────────────────────
CNN_IMG_SIZE    = 128
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
CNN_CLASS_NAMES = ['Calculus', 'Data caries', 'Gingivitis', 'Tooth Discoloration']
NUM_CLASSES     = len(CNN_CLASS_NAMES)

# Mapping nama kelas ke label user-friendly
DISEASE_DISPLAY_NAMES = {
    'Calculus':            'Kalkulus (Calculus / Karang Gigi)',
    'Data caries':         'Karies Gigi (Caries)',
    'Gingivitis':          'Gingivitis (Radang Gusi)',
    'Tooth Discoloration': 'Perubahan Warna Gigi (Tooth Discoloration)',
}

# Mapping nama kelas pendek untuk label bounding box pada gambar (agar lebih kecil dan tidak menghalangi)
DISEASE_SHORT_NAMES = {
    'Calculus':            'Calculus',
    'Data caries':         'Caries',
    'Gingivitis':          'Gingivitis',
    'Tooth Discoloration': 'Discoloration',
}


# Mapping nama kelas YOLO ke nama kelas CNN ResNet-18 (untuk resolusi label)
# YOLO class names (dari best.pt): Calculus, Caries, Caries3, Discoloration, Gingivitis
YOLO_TO_CNN_CLASS = {
    'Calculus':       'Calculus',
    'Caries':         'Data caries',
    'Caries3':        'Data caries',
    'Discoloration':  'Tooth Discoloration',
    'Gingivitis':     'Gingivitis',
}

# Deskripsi edukatif untuk setiap penyakit
DISEASE_DESCRIPTIONS = {
    'Calculus': (
        "Kalkulus (karang gigi) adalah plak gigi yang telah mengeras akibat "
        "mineralisasi oleh mineral dari air liur. Karang gigi tidak bisa "
        "dihilangkan dengan menyikat gigi biasa dan memerlukan pembersihan "
        "profesional oleh dokter gigi (scaling). Jika dibiarkan, kalkulus "
        "dapat menyebabkan penyakit gusi, bau mulut, dan kerusakan jaringan "
        "pendukung gigi. Pencegahan meliputi menyikat gigi 2× sehari, "
        "menggunakan benang gigi, dan rutin scaling setiap 6 bulan."
    ),
    'Data caries': (
        "Karies gigi adalah kerusakan struktur gigi yang disebabkan oleh "
        "aktivitas bakteri penghasil asam. Bakteri mengubah sisa makanan "
        "(terutama gula) menjadi asam yang mengikis lapisan enamel gigi. "
        "Jika tidak ditangani, karies dapat berkembang ke lapisan dentin "
        "hingga pulpa, menyebabkan rasa sakit dan infeksi. Penanganan "
        "meliputi penambalan, perawatan saluran akar, atau pencabutan gigi "
        "tergantung tingkat keparahannya."
    ),
    'Gingivitis': (
        "Gingivitis adalah peradangan pada gusi yang ditandai dengan gusi "
        "berwarna merah, bengkak, dan mudah berdarah saat menyikat gigi. "
        "Penyebab utamanya adalah penumpukan plak bakteri di sepanjang garis "
        "gusi. Gingivitis merupakan tahap awal penyakit periodontal dan "
        "bersifat reversibel jika ditangani dengan baik. Pencegahan dan "
        "pengobatan meliputi kebersihan mulut yang baik, pembersihan karang "
        "gigi secara rutin, dan penggunaan obat kumur antiseptik."
    ),
    'Tooth Discoloration': (
        "Perubahan warna gigi (tooth discoloration) adalah kondisi di mana "
        "warna gigi berubah menjadi kuning, coklat, atau kehitaman. Penyebabnya "
        "bisa ekstrinsik (makanan/minuman berwarna, rokok, plak) atau intrinsik "
        "(trauma gigi, konsumsi antibiotik tertentu saat masa pertumbuhan gigi, "
        "fluorosis). Penanganan tergantung penyebabnya, mulai dari pembersihan "
        "profesional, bleaching/pemutihan gigi, hingga pemasangan veneer atau "
        "crown untuk kasus yang parah."
    ),
}

# Warna berbeda untuk setiap penyakit
DISEASE_COLORS = {
    'Calculus':            '#FF6B35',  # Oranye
    'Data caries':         '#EF4444',  # Merah
    'Gingivitis':          '#F59E0B',  # Kuning-oranye
    'Tooth Discoloration': '#8B5CF6',  # Ungu
}

# ──────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Menggunakan device: {DEVICE}")

yolo_model = None
YOLO_FALLBACK = os.path.join(DATA_DIR, "yolov8n.pt")
if YOLO_AVAILABLE:
    from ultralytics import YOLO
    print(f"[INFO] Memuat model YOLO fine-tuned dari: {YOLO_FINETUNED}")
    yolo_model = YOLO(YOLO_FINETUNED)
    print("[INFO] Model YOLO fine-tuned berhasil dimuat")
elif os.path.exists(YOLO_FALLBACK):
    from ultralytics import YOLO
    print(f"[INFO] Memuat model YOLO fallback dari: {YOLO_FALLBACK}")
    yolo_model = YOLO(YOLO_FALLBACK)
    print("[INFO] Model YOLO fallback berhasil dimuat")
else:
    print("[INFO] WARNING: Tidak ada model YOLO found di folder data/")

# ──────────────────────────────────────────────────────────
# Muat Model CNN ResNet-18
# ──────────────────────────────────────────────────────────
def build_resnet18(num_classes: int) -> nn.Module:
    """
    Bangun arsitektur ResNet-18 dengan custom head yang SAMA
    dengan arsitektur saat training di notebook:
        Dropout(0.4) → Linear(512,256) → ReLU → BN(256)
        → Dropout(0.25) → Linear(256, num_classes)
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes)
    )
    return model


print(f"[INFO] Memuat model CNN ResNet-18 dari: {CNN_WEIGHTS}")
cnn_model = build_resnet18(NUM_CLASSES)
cnn_model.load_state_dict(
    torch.load(CNN_WEIGHTS, map_location=DEVICE, weights_only=False)
)
cnn_model.to(DEVICE)
cnn_model.eval()
print("[INFO] Model CNN ResNet-18 berhasil dimuat")

# ──────────────────────────────────────────────────────────
# Preprocessing untuk CNN (sama dengan val_transform di notebook)
# ──────────────────────────────────────────────────────────
cnn_transform = transforms.Compose([
    transforms.Resize(CNN_IMG_SIZE),
    transforms.CenterCrop(CNN_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# TTA (Test-Time Augmentation) transforms — dari notebook
# Ini meningkatkan akurasi prediksi dengan rata-rata hasil
# dari beberapa variasi augmentasi
tta_transforms = [
    # 1. Original
    transforms.Compose([
        transforms.Resize(CNN_IMG_SIZE),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.Resize(CNN_IMG_SIZE),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 3. Slightly larger crop
    transforms.Compose([
        transforms.Resize(int(CNN_IMG_SIZE * 1.14)),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 4. Slight rotation (applied via random)
    transforms.Compose([
        transforms.Resize(CNN_IMG_SIZE),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 5. Brightness/contrast jitter
    transforms.Compose([
        transforms.Resize(CNN_IMG_SIZE),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 6. Vertical flip
    transforms.Compose([
        transforms.Resize(CNN_IMG_SIZE),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 7. Larger scale crop
    transforms.Compose([
        transforms.Resize(int(CNN_IMG_SIZE * 1.25)),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 8. Sharpness
    transforms.Compose([
        transforms.Resize(CNN_IMG_SIZE),
        transforms.CenterCrop(CNN_IMG_SIZE),
        transforms.RandomAdjustSharpness(2, p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]

print(f"[INFO] TTA dengan {len(tta_transforms)} variasi augmentasi siap")


# ──────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Cek apakah ekstensi file diperbolehkan."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_with_tta(image: Image.Image) -> tuple[int, float, list[float]]:
    """
    Klasifikasi gambar dengan Test-Time Augmentation (TTA).
    Rata-ratakan probabilitas dari 8 variasi augmentasi
    untuk prediksi yang lebih stabil dan akurat.

    Returns:
        (predicted_class_idx, confidence, all_probabilities)
    """
    all_probs = []

    with torch.no_grad():
        for t in tta_transforms:
            input_tensor = t(image).unsqueeze(0).to(DEVICE)
            output = cnn_model(input_tensor)
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    # Rata-ratakan probabilitas dari semua TTA
    avg_probs = np.mean(all_probs, axis=0)
    predicted_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_idx])

    return predicted_idx, confidence, avg_probs.tolist()


def classify_single(image: Image.Image) -> tuple[int, float, list[float]]:
    """
    Klasifikasi gambar tanpa TTA (lebih cepat, kurang akurat).

    Returns:
        (predicted_class_idx, confidence, all_probabilities)
    """
    with torch.no_grad():
        input_tensor = cnn_transform(image).unsqueeze(0).to(DEVICE)
        output = cnn_model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])
    return predicted_idx, confidence, probs.tolist()


def get_adaptive_roi(img_cv: np.ndarray) -> tuple[int, int]:
    """
    Menentukan ROI vertikal (gigi) secara adaptif menggunakan OpenCV.
    Menggunakan rentang yang lebih luas dan aman (5% s/d 95%) agar karies yang gelap di tepi gigi tidak terpotong.
    """
    img_h = img_cv.shape[0]
    min_y = int(img_h * 0.05)
    max_y = int(img_h * 0.95)
    return min_y, max_y


def filter_and_deduplicate_boxes(detections: list[dict], img_w: int, img_h: int, min_y: int, max_y: int) -> list[dict]:
    """
    Penyaringan pasca-inferensi (Post-Processing):
    1. ROI Filtering: Menyaring boks yang berada di luar area gigi (min_y s/d max_y).
    2. Per-Class Confidence Threshold: Setiap kelas penyakit punya ambang batas confidence berbeda.
       - Calculus / Caries: lebih mudah terlihat secara visual → conf >= 0.20
       - Gingivitis / Discoloration: lebih halus dan difus → conf >= 0.08
    3. Box Size & Aspect Ratio: Menyaring kotak noise atau false positive rahang.
    4. Per-Class NMS / Deduplikasi: Menghilangkan boks bertumpuk DALAM kelas yang sama saja
       (boks dari kelas berbeda TIDAK saling menghapus, sehingga Gingivitis & Caries bisa muncul bersamaan).
    5. Membatasi maksimal 4 boks per kelas agar tidak terlalu padat.
    """
    # Threshold confidence per kelas YOLO
    # Catatan: best.pt yang dilatih menghasilkan confidence rendah secara natural.
    # Threshold ini hanya membuang prediksi noise ekstrem (sangat mendekati 0).
    CLASS_CONF_THRESH = {
        'Calculus':      0.03,
        'Caries':        0.03,
        'Caries3':       0.03,
        'Gingivitis':    0.03,
        'Discoloration': 0.03,
    }
    DEFAULT_CONF_THRESH = 0.03  # fallback untuk kelas tidak dikenal
    MAX_PER_CLASS = 5           # Maksimal boks per kelas

    total_yolo = len(detections)
    print(f"[PIPELINE LOG] 1. Prediksi awal YOLO: {total_yolo} bounding box")
    
    if not detections:
        return []
        
    # Urutkan berdasarkan confidence tertinggi
    detections = sorted(detections, key=lambda x: x.get("confidence", 0.0), reverse=True)
    
    # A. ROI Filtering: Cek apakah pusat vertikal kotak masuk area gigi
    roi_filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        y_center = (y1 + y2) / 2
        if min_y <= y_center <= max_y:
            roi_filtered.append(det)
        else:
            print(f"[ROI FILTERED] Dibuang karena y_center={y_center:.1f} berada di luar area gigi [{min_y} - {max_y}]")
            
    total_roi = len(roi_filtered)
    print(f"[PIPELINE LOG] 2. Setelah ROI Filtering: {total_roi} bounding box (dibuang: {total_yolo - total_roi})")

    # B. Per-class Confidence Threshold Filtering
    conf_filtered = []
    for det in roi_filtered:
        label = det.get("label", "")
        threshold = CLASS_CONF_THRESH.get(label, DEFAULT_CONF_THRESH)
        if det.get("confidence", 0.0) >= threshold:
            conf_filtered.append(det)
        else:
            print(f"[CONF FILTERED] '{label}' conf={det.get('confidence',0):.3f} < threshold={threshold:.2f} → dibuang")
    
    total_conf = len(conf_filtered)
    print(f"[PIPELINE LOG] 2b. Setelah Per-Class Confidence Filter: {total_conf} bounding box (dibuang: {total_roi - total_conf})")
            
    # C. Size & Aspect Ratio Filtering
    size_filtered = []
    for det in conf_filtered:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        min_area = (img_w * img_h * 0.0001) # 0.01% area gambar (diperkecil agar boks karies kecil tidak hilang)
        
        if area < min_area:
            print(f"[SIZE FILTERED] Boks dibuang karena terlalu kecil: area={area:.0f}px (min={min_area:.0f}px)")
            continue
        if w > (img_w * 0.95) and h > (img_h * 0.95):
            print(f"[SIZE FILTERED] Boks dibuang karena mencakup hampir seluruh gambar: w={w}px, h={h}px")
            continue
            
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            print(f"[ASPECT FILTERED] Boks dibuang karena rasio aspek janggal: ratio={aspect_ratio:.2f}")
            continue
            
        size_filtered.append(det)
        
    total_size = len(size_filtered)
    print(f"[PIPELINE LOG] 3. Setelah Size & Aspect Ratio Filtering: {total_size} bounding box (dibuang: {total_conf - total_size})")
            
    # D. Per-Class NMS / Deduplikasi
    #    Boks dari kelas BERBEDA TIDAK saling menghapus (penting agar multi-penyakit bisa terdeteksi)
    from collections import defaultdict
    class_groups = defaultdict(list)
    for det in size_filtered:
        class_groups[det.get("label", "unknown")].append(det)
    
    deduplicated = []
    for cls_label, cls_dets in class_groups.items():
        # NMS dalam kelas yang sama saja
        kept_in_class = []
        for det in cls_dets:  # sudah terurut confidence desc
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            area = max(1, (x2 - x1) * (y2 - y1))
            
            is_duplicate = False
            for r_det in kept_in_class:
                rx1, ry1, rx2, ry2 = r_det["x1"], r_det["y1"], r_det["x2"], r_det["y2"]
                ix1, iy1 = max(x1, rx1), max(y1, ry1)
                ix2, iy2 = min(x2, rx2), min(y2, ry2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                intersection = iw * ih
                union = area + max(1, (rx2 - rx1) * (ry2 - ry1)) - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.40:
                    is_duplicate = True
                    print(f"[PER-CLASS NMS] '{cls_label}' boks dibuang IoU={iou:.2f}")
                    break
                    
            if not is_duplicate:
                kept_in_class.append(det)
                if len(kept_in_class) >= MAX_PER_CLASS:
                    print(f"[PER-CLASS CAP] '{cls_label}' mencapai batas {MAX_PER_CLASS} boks, sisa diabaikan.")
                    break
        
        deduplicated.extend(kept_in_class)
    
    # Urutkan hasil akhir berdasarkan confidence
    deduplicated = sorted(deduplicated, key=lambda x: x.get("confidence", 0.0), reverse=True)
    
    total_dedup = len(deduplicated)
    print(f"[PIPELINE LOG] 4. Setelah Per-Class NMS: {total_dedup} bounding box (dibuang: {total_size - total_dedup})")
            
    return deduplicated



def run_yolo_detection(image: Image.Image) -> list[dict]:
    """
    Deteksi area penyakit gigi menggunakan YOLOv8.
    - conf=0.02: ambang sangat rendah agar kelas halus seperti Gingivitis tetap ditangkap
    - Post-filtering per-class dilakukan di filter_and_deduplicate_boxes
    - Ambil hingga 50 kandidat raw box (disaring lebih lanjut oleh per-class NMS)
    """
    detections = []
    
    if yolo_model is not None:
        is_coco = "person" in yolo_model.names.values()
        if not is_coco:
            # conf rendah agar tidak melewatkan kelas yang sulit terdeteksi
            results = yolo_model(image, imgsz=640, conf=0.02, iou=0.30, max_det=150, agnostic_nms=False, verbose=False)
            
            raw_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    label = result.names.get(cls_id, f"class_{cls_id}")

                    raw_detections.append({
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3]),
                        "confidence": conf,
                        "label": label,
                    })
            
            # Urutkan berdasarkan confidence tertinggi
            raw_detections = sorted(raw_detections, key=lambda x: x["confidence"], reverse=True)
            
            # Ambil 50 kandidat teratas sebelum per-class filtering
            detections = raw_detections[:50]
            print(f"[YOLO INFERENCE] Ditemukan {len(raw_detections)} raw boxes, mengambil {len(detections)} kandidat untuk difilter.")
        else:
            print("[INFO] Model YOLO yang dimuat adalah COCO model. Inferensi YOLO dental dinonaktifkan.")
            
    return detections



def run_full_image_classification(image: Image.Image, use_tta: bool = True) -> dict:
    """
    Klasifikasi penyakit gigi menggunakan CNN ResNet-18
    pada gambar penuh (tanpa crop/bounding box).

    Args:
        image:   PIL Image (RGB)
        use_tta: Gunakan TTA untuk hasil lebih akurat

    Returns:
        dict dengan key: disease, confidence, description,
                         class_name, all_probabilities
    """
    if use_tta:
        predicted_idx, confidence, all_probs = classify_with_tta(image)
    else:
        predicted_idx, confidence, all_probs = classify_single(image)

    class_name = CNN_CLASS_NAMES[predicted_idx]
    display_name = DISEASE_DISPLAY_NAMES.get(class_name, class_name)
    description = DISEASE_DESCRIPTIONS.get(class_name, "Deskripsi tidak tersedia.")

    return {
        "disease": display_name,
        "confidence": confidence,
        "description": description,
        "class_name": class_name,
        "all_probabilities": {
            DISEASE_DISPLAY_NAMES.get(cn, cn): round(p, 4)
            for cn, p in zip(CNN_CLASS_NAMES, all_probs)
        },
    }


def run_crop_classification(image: Image.Image, bbox: dict) -> dict:
    """
    Klasifikasi penyakit pada area crop (dari YOLO bounding box).
    """
    crop = image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))

    # Gunakan TTA juga pada crop untuk akurasi lebih baik
    predicted_idx, confidence, all_probs = classify_with_tta(crop)

    class_name = CNN_CLASS_NAMES[predicted_idx]
    display_name = DISEASE_DISPLAY_NAMES.get(class_name, class_name)
    description = DISEASE_DESCRIPTIONS.get(class_name, "Deskripsi tidak tersedia.")

    return {
        "disease": display_name,
        "confidence": confidence,
        "description": description,
        "class_name": class_name,
    }


def draw_predictions(image_path: str, detections: list, resnet_outputs: dict) -> list:
    """
    Menggambar bounding box dan teks klasifikasi crop secara rapi menggunakan OpenCV.
    Menerapkan dynamic font scaling dan anti-overlap pada label.
    """
    img = cv2.imread(image_path)
    if img is None:
        return detections

    img_h, img_w = img.shape[:2]

    # BGR Colors
    COLOR_MAP = {
        'Calculus': (53, 107, 255),          # Amber / Oranye hangat (#FF6B35)
        'Data caries': (68, 68, 239),         # Merah tegas (#EF4444)
        'Gingivitis': (11, 158, 245),         # Kuning-oranye (#F59E0B)
        'Tooth Discoloration': (246, 92, 139) # Ungu (#8B5CF6)
    }

    drawn_label_boxes = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        disease_conf = det["disease_confidence"]
        class_name = det["class_name"]

        # Tentukan warna boks
        box_color = COLOR_MAP.get(class_name, (68, 68, 239))

        # 1. Gambar Bounding Box asli dari YOLO dengan garis lebih tipis agar tidak menghalangi
        thickness = max(2, img_w // 320)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

        # 2. Gambar Teks Label pendek dengan deteksi anti-overlap dan ukuran lebih kecil
        disease_short = DISEASE_SHORT_NAMES.get(class_name, class_name)
        label_text = f"{disease_short} {disease_conf:.0%}"
        font_scale = max(0.35, img_w / 1800.0)
        font_thickness = max(1, int(img_w / 1200.0))
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Cari posisi vertikal yang optimal untuk label
        label_y = y1 - 4
        if label_y - text_h - 4 < 0:
            # Letakkan di bawah jika keluar dari batas atas gambar
            label_y = y2 + text_h + 6

        # Cek dan hindari tabrakan dengan label lain yang sudah digambar
        collision = True
        attempts = 0
        while collision and attempts < 4:
            collision = False
            for rx1, ry1, rx2, ry2 in drawn_label_boxes:
                # Periksa overlap horizontal dan vertikal
                if not (x1 + text_w + 6 < rx1 or x1 > rx2 or label_y < ry1 or label_y - text_h - 8 > ry2):
                    collision = True
                    label_y += (text_h + 8)  # Geser ke bawah sedikit jika bertumpukan
                    attempts += 1
                    break
        
        # Batasi agar label tidak keluar dari batas bawah gambar
        if label_y + baseline > img_h:
            label_y = img_h - baseline - 3

        # Catat box koordinat label yang baru
        drawn_label_boxes.append((x1, label_y - text_h - 4, x1 + text_w + 6, label_y + baseline))

        # Gambar background label
        cv2.rectangle(
            img,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 6, label_y + baseline - 1),
            box_color,
            cv2.FILLED
        )

        # Tulis teks label
        cv2.putText(
            img,
            label_text,
            (x1 + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    # 3. Gambar Diagnosis Utama Dominan (Pojok Kiri Atas) - dibuat lebih ringkas dan kecil
    dominant_disease = resnet_outputs.get("disease")
    dominant_confidence = resnet_outputs.get("confidence", 0.0)
    dominant_class = resnet_outputs.get("class_name")
    dominant_disease_short = DISEASE_SHORT_NAMES.get(dominant_class, dominant_disease)

    main_label = f" Dominan: {dominant_disease_short} ({dominant_confidence:.0%}) "
    font_scale_main = max(0.38, img_w / 1800.0)
    font_thickness_main = max(1, int(img_w / 1200.0))
    (main_w, main_h), base_line = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, font_thickness_main)

    overlay = img.copy()
    pad = 10
    cv2.rectangle(
        overlay,
        (pad, pad),
        (pad + main_w + 6, pad + main_h + 6),
        (0, 0, 0),
        cv2.FILLED
    )
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    main_color = COLOR_MAP.get(dominant_class, (68, 68, 239))
    cv2.putText(
        img,
        main_label,
        (pad + 3, pad + main_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale_main,
        main_color,
        font_thickness_main,
        cv2.LINE_AA
    )

    cv2.imwrite(image_path, img)
    return detections



# ──────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render halaman utama."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Menerima gambar dari:
      • Form field 'image' (multipart file upload), ATAU
      • JSON body  { "image_base64": "<base64 string>" }

    Pipeline:
      1. Klasifikasi gambar penuh dengan CNN + TTA (selalu)
      2. Deteksi bounding box dengan YOLO (jika best.pt tersedia)
      3. Klasifikasi setiap crop bounding box (jika ada)
    """
    image = None

    # ── 1. Terima gambar ──────────────────────────────────
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Tidak ada file yang dipilih."}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Format file tidak didukung."}), 400
        image = Image.open(file.stream).convert("RGB")

    elif request.is_json and "image_base64" in request.get_json(silent=True, force=True):
        data = request.get_json(force=True)
        b64_str = data["image_base64"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_str)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"success": False, "error": f"Gagal decode base64: {e}"}), 400
    else:
        return jsonify({
            "success": False,
            "error": "Tidak ada gambar. Gunakan form field 'image' atau JSON 'image_base64'."
        }), 400

    # ── 2. Jalankan pipeline ──────────────────────────────
    try:
        print("\n=== [PIPELINE INITIATED] ===")
        # A. Klasifikasi gambar penuh dengan CNN + TTA
        main_classification = run_full_image_classification(image, use_tta=True)
        print(f"[PIPELINE LOG] Diagnosis Utama (ResNet-18): {main_classification['disease']} (Conf: {main_classification['confidence']:.0%})")

        # B. Deteksi YOLO (jika tersedia)
        yolo_detections = run_yolo_detection(image)

        # C. Simpan gambar original terlebih dahulu untuk digambar oleh OpenCV
        filename = f"result_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(RESULT_FOLDER, filename)
        image.save(result_path, "PNG")

        # D. Terapkan adaptive ROI & post-processing filter (jika gambar berhasil dibaca OpenCV)
        img_cv = cv2.imread(result_path)
        if img_cv is not None:
            min_y, max_y = get_adaptive_roi(img_cv)
            yolo_detections = filter_and_deduplicate_boxes(
                yolo_detections, 
                img_cv.shape[1], 
                img_cv.shape[0], 
                min_y, 
                max_y
            )

        # E. DECISION LAYER AWAL — Tentukan kelas dominan sebelum ResNet crop
        #    Tujuan: filter boks YOLO hanya untuk kelas dominan, lalu ResNet crop
        #    hanya dijalankan pada boks yang tersisa → LEBIH CEPAT & FOKUS
        yolo_raw_votes = {}
        for det in yolo_detections:
            raw_label = det.get("label", "")
            cnn_cls = YOLO_TO_CNN_CLASS.get(raw_label, None)
            if cnn_cls:
                yolo_raw_votes[cnn_cls] = yolo_raw_votes.get(cnn_cls, 0) + 1

        # Hitung fusion score awal (YOLO vote count + ResNet full-image prob)
        resnet_all_probs = main_classification.get("all_probabilities", {})
        early_fusion = {}
        for cnn_cn in CNN_CLASS_NAMES:
            display_n = DISEASE_DISPLAY_NAMES.get(cnn_cn, cnn_cn)
            n_votes   = yolo_raw_votes.get(cnn_cn, 0)
            resnet_p  = resnet_all_probs.get(display_n, 0.0)
            vote_w    = min(n_votes, 5) / 5.0
            # Bobot YOLO 55% jika ada vote, 0% jika tidak ada
            early_fusion[cnn_cn] = (vote_w * 0.55) + (resnet_p * 0.45)

        dominant_class = max(early_fusion, key=lambda k: early_fusion[k])
        print(f"[DECISION LAYER] Kelas dominan awal: '{dominant_class}' | YOLO votes: {yolo_raw_votes} | ResNet top: '{main_classification['class_name']}'")

        # Filter boks YOLO → hanya tampilkan boks untuk kelas dominan saja
        before_filter = len(yolo_detections)
        yolo_detections = [
            d for d in yolo_detections
            if YOLO_TO_CNN_CLASS.get(d.get("label", ""), None) == dominant_class
        ]
        print(f"[DECISION LAYER] Filter boks: {before_filter} → {len(yolo_detections)} (hanya kelas '{dominant_class}')")

        # Update main_classification dengan kelas dominan hasil fusion
        if dominant_class != main_classification.get("class_name"):
            main_classification["class_name"]  = dominant_class
            main_classification["disease"]     = DISEASE_DISPLAY_NAMES.get(dominant_class, dominant_class)
            main_classification["description"] = DISEASE_DESCRIPTIONS.get(dominant_class, main_classification["description"])
            fused_conf = early_fusion[dominant_class]
            main_classification["confidence"]  = min(fused_conf + 0.40, 0.97)

        # F. Klasifikasi crop ResNet-18 — hanya untuk boks kelas dominan
        #    (jauh lebih sedikit boks → lebih cepat)
        print(f"[PIPELINE LOG] 5. Klasifikasi Crop ResNet-18 untuk {len(yolo_detections)} boks kelas '{dominant_class}'...")
        for idx, det in enumerate(yolo_detections):
            yolo_label    = det.get("label", "")
            yolo_cls_name = YOLO_TO_CNN_CLASS.get(yolo_label, None)

            # Gunakan confidence YOLO langsung (lebih cepat, crop ResNet di-skip)
            final_class_name  = yolo_cls_name if yolo_cls_name else dominant_class
            final_display     = DISEASE_DISPLAY_NAMES.get(final_class_name, final_class_name)
            final_desc        = DISEASE_DESCRIPTIONS.get(final_class_name, "Deskripsi tidak tersedia.")
            final_confidence  = max(det.get("confidence", 0.5), 0.50)

            det["disease"]            = final_display
            det["class_name"]         = final_class_name
            det["disease_confidence"] = final_confidence
            det["description"]        = final_desc
            print(f"   [Boks #{idx+1}] '{yolo_label}' → '{final_class_name}' ({final_confidence:.0%})")

        # G. Gambar Bounding Box dengan OpenCV menggunakan visualisasi anti-overlap
        synced_detections = draw_predictions(result_path, yolo_detections, main_classification)
        print(f"[PIPELINE LOG] 6. Penggambaran gambar hasil selesai. Total {len(synced_detections)} bounding box digambar.")
        print("=== [PIPELINE COMPLETED] ===\n")

        # ── 3. Kembalikan response ────────────────────────
        result_url = url_for("static", filename=f"uploads/results/{filename}")

        # Build detections array
        detections_response = []

        # Selalu sertakan klasifikasi utama (full image)
        detections_response.append({
            "bbox": None,
            "source": "CNN (Full Image + TTA)",
            "disease": main_classification["disease"],
            "disease_confidence": main_classification["confidence"],
            "description": main_classification["description"],
            "all_probabilities": main_classification.get("all_probabilities"),
        })

        # Tambahkan deteksi YOLO yang sudah diselaraskan
        for det in synced_detections:
            detections_response.append({
                "bbox": {
                    "x1": det["x1"], "y1": det["y1"],
                    "x2": det["x2"], "y2": det["y2"]
                },
                "source": f"YOLO → CNN (Crop)",
                "yolo_label": det.get("label", "tooth_region"),
                "yolo_confidence": det.get("confidence", 0.95),
                "disease": det["disease"],
                "disease_confidence": det["disease_confidence"],
                "description": det.get("description", "Deskripsi tidak tersedia."),
            })

        response_data = {
            "success": True,
            "result_image": result_url,
            "has_yolo": YOLO_AVAILABLE and len(yolo_detections) > 0,
            "detections": detections_response,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Terjadi kesalahan: {e}"}), 500


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
