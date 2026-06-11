from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import cv2
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_katarak_obb.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
HISTORY_CSV = os.path.join(DATA_FOLDER, "history.csv")
HISTORY_HEADER = [
    "id", "tanggal", "status", "risk_level", "confidence", "cataract_confidence",
    "normal_confidence", "jumlah_objek", "quality_status", "uploaded_image",
    "result_image", "report_url"
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}. Letakkan file best_katarak_obb.pt di folder models/")

model = YOLO(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_class_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def is_normal_class(name):
    n = normalize_class_name(name)
    normal_keywords = [
        "normal",
        "sehat",
        "healthy",
        "non cataract",
        "noncataract",
        "non katarak",
        "nonkatarak",
        "no cataract",
        "no katarak",
        "not cataract",
        "not katarak",
        "without cataract",
        "without katarak",
        "negative"
    ]
    return any(keyword in n for keyword in normal_keywords)


def is_cataract_class(name):
    n = normalize_class_name(name)
    cataract_keywords = [
        "cataract",
        "katarak",
        "kataract",
        "positive"
    ]
    return any(keyword in n for keyword in cataract_keywords) and not is_normal_class(name)


def classify_detection_result(detections):
    """
    BUG FIX:
    Versi sebelumnya memakai confidence tertinggi dari semua box.
    Itu salah jika class yang terdeteksi adalah NORMAL.

    Versi ini:
    - Jika ada class cataract/katarak, status = TERINDIKASI KATARAK.
    - Jika hanya ada class normal, status = NORMAL / TIDAK TERINDIKASI.
    - Risk level dihitung dari confidence class katarak saja.
    - Confidence normal tidak dipakai sebagai confidence katarak.
    """
    cataract_detections = [d for d in detections if d["is_cataract"]]
    normal_detections = [d for d in detections if d["is_normal"]]
    unknown_detections = [d for d in detections if not d["is_cataract"] and not d["is_normal"]]

    if cataract_detections:
        max_cataract = max(d["confidence"] for d in cataract_detections)
        risk_level, risk_class = get_risk_level(max_cataract)
        return {
            "status": "TERINDIKASI KATARAK",
            "summary_label": "Katarak",
            "risk_level": risk_level,
            "risk_class": risk_class,
            "confidence": round(max_cataract, 2),
            "cataract_confidence": round(max_cataract, 2),
            "normal_confidence": round(max((d["confidence"] for d in normal_detections), default=0), 2),
            "main_detections": cataract_detections,
            "decision_reason": "Model mendeteksi class katarak. Risiko dihitung dari confidence class katarak."
        }

    if normal_detections:
        max_normal = max(d["confidence"] for d in normal_detections)
        return {
            "status": "TIDAK TERINDIKASI KATARAK",
            "summary_label": "Normal",
            "risk_level": "Risiko Rendah",
            "risk_class": "low",
            "confidence": round(max_normal, 2),
            "cataract_confidence": 0,
            "normal_confidence": round(max_normal, 2),
            "main_detections": normal_detections,
            "decision_reason": "Model hanya mendeteksi class normal. Confidence ini adalah confidence normal, bukan confidence katarak."
        }

    if unknown_detections:
        max_unknown = max(d["confidence"] for d in unknown_detections)
        return {
            "status": "CLASS TIDAK DIKENALI",
            "summary_label": "Perlu Cek Label",
            "risk_level": "Perlu Verifikasi",
            "risk_class": "medium",
            "confidence": round(max_unknown, 2),
            "cataract_confidence": 0,
            "normal_confidence": 0,
            "main_detections": unknown_detections,
            "decision_reason": "Ada box terdeteksi, tetapi nama class tidak mengandung normal/sehat atau cataract/katarak. Cek class names model."
        }

    return {
        "status": "TIDAK ADA OBJEK TERDETEKSI",
        "summary_label": "Tidak Terdeteksi",
        "risk_level": "Risiko Rendah",
        "risk_class": "low",
        "confidence": 0,
        "cataract_confidence": 0,
        "normal_confidence": 0,
        "main_detections": [],
        "decision_reason": "Tidak ada bounding box yang lolos threshold confidence."
    }


def get_risk_level(cataract_confidence):
    # Risk level HANYA untuk confidence class katarak.
    if cataract_confidence >= 70:
        return "Risiko Tinggi", "high"
    if cataract_confidence >= 40:
        return "Risiko Sedang", "medium"
    return "Risiko Rendah", "low"


def get_advice(status, risk_level):
    if status == "TERINDIKASI KATARAK":
        if risk_level == "Risiko Tinggi":
            return "Disarankan segera melakukan pemeriksaan lanjutan ke dokter spesialis mata."
        if risk_level == "Risiko Sedang":
            return "Disarankan berkonsultasi ke dokter mata, terutama jika ada keluhan penglihatan."
        return "Terdapat indikasi rendah. Ambil ulang foto yang lebih jelas atau konsultasi jika ada keluhan."
    if status == "TIDAK TERINDIKASI KATARAK":
        return "Hasil AI menunjukkan class normal pada gambar ini. Tetap periksa ke dokter mata jika ada keluhan."
    if status == "CLASS TIDAK DIKENALI":
        return "Nama class model belum dikenali sistem. Periksa nama class pada model/dataset."
    return "Tidak ada objek yang terdeteksi. Coba gunakan foto yang lebih jelas dan pencahayaan cukup."


def get_action_plan(status, risk_level):
    if status == "TERINDIKASI KATARAK":
        if risk_level == "Risiko Tinggi":
            return [
                "Jangan gunakan hasil AI sebagai diagnosis final.",
                "Simpan hasil pemeriksaan ini.",
                "Segera cari dokter spesialis mata atau rumah sakit mata terdekat.",
            ]
        return [
            "Ambil ulang foto jika gambar kurang jelas.",
            "Pantau keluhan seperti buram, silau, atau sulit melihat malam hari.",
            "Pertimbangkan konsultasi ke dokter mata.",
        ]

    if status == "TIDAK TERINDIKASI KATARAK":
        return [
            "Gunakan foto dengan pencahayaan baik untuk hasil lebih stabil.",
            "Ulangi pemeriksaan jika ada perubahan kondisi mata.",
            "Konsultasi ke dokter bila tetap ada keluhan.",
        ]

    return [
        "Cek nama class pada dataset/model.",
        "Pastikan class katarak bernama cataract/katarak dan class normal bernama normal/sehat.",
        "Ambil ulang foto atau turunkan/naikkan threshold jika diperlukan.",
    ]



def make_voice_text(status, risk_level, cataract_confidence, normal_confidence, advice):
    """
    Teks suara dibuat di backend agar frontend tidak salah menafsirkan hasil.
    - Normal: tidak menyebut terdeteksi katarak.
    - Katarak: menyebut confidence katarak.
    - Unknown/no object: memberi instruksi ambil ulang/cek label.
    """
    cataract_round = int(round(float(cataract_confidence or 0)))
    normal_round = int(round(float(normal_confidence or 0)))

    if status == "TERINDIKASI KATARAK":
        return (
            f"Hasil analisis menunjukkan indikasi katarak "
            f"dengan confidence katarak {cataract_round} persen. "
            f"Tingkat risiko {risk_level}. {advice}"
        )

    if status == "TIDAK TERINDIKASI KATARAK":
        return (
            f"Hasil analisis menunjukkan mata normal "
            f"dengan confidence normal {normal_round} persen. "
            f"Tidak ada indikasi katarak pada gambar ini. {advice}"
        )

    if status == "TIDAK ADA OBJEK TERDETEKSI":
        return (
            "Tidak ada objek mata yang terdeteksi dengan jelas pada gambar ini. "
            "Silakan ambil ulang foto dengan pencahayaan yang lebih baik."
        )

    return (
        "Hasil belum dapat dibacakan dengan pasti karena class model tidak dikenali sistem. "
        "Silakan cek nama class pada model."
    )

def analyze_image_quality(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {
            "brightness": "Tidak terbaca",
            "sharpness": "Tidak terbaca",
            "resolution": "Tidak terbaca",
            "quality_score": 0,
            "quality_status": "Gambar tidak dapat dibaca",
            "recommendation": "Upload ulang gambar dengan format JPG/PNG yang valid."
        }

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_value = float(np.mean(gray))
    sharpness_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness = "Cukup"
    if brightness_value < 60:
        brightness = "Terlalu gelap"
    elif brightness_value > 210:
        brightness = "Terlalu terang"

    sharpness = "Baik" if sharpness_value >= 80 else "Cenderung buram"
    resolution = "Baik" if min(w, h) >= 400 else "Terlalu kecil"

    score = 0
    score += 40 if brightness == "Cukup" else 15
    score += 40 if sharpness == "Baik" else 15
    score += 20 if resolution == "Baik" else 8

    quality_status = "Baik" if score >= 80 else "Cukup" if score >= 55 else "Perlu diperbaiki"

    recs = []
    if brightness != "Cukup":
        recs.append("atur pencahayaan")
    if sharpness != "Baik":
        recs.append("ambil foto lebih fokus")
    if resolution != "Baik":
        recs.append("gunakan gambar resolusi lebih besar")

    recommendation = "Kualitas foto layak untuk dianalisis." if not recs else "Sebaiknya " + ", ".join(recs) + "."

    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "resolution": resolution,
        "quality_score": score,
        "quality_status": quality_status,
        "recommendation": recommendation
    }


def ensure_history_schema():
    """Pastikan CSV riwayat memakai header terbaru agar tidak bug setelah update versi."""
    if not os.path.exists(HISTORY_CSV):
        return

    with open(HISTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        rows = list(reader)

    if old_fieldnames == HISTORY_HEADER:
        return

    normalized_rows = []
    for row in rows:
        normalized_rows.append({key: row.get(key, "") for key in HISTORY_HEADER})

    write_history(normalized_rows)


def save_history(row):
    ensure_history_schema()
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_HEADER)
        if not file_exists:
            writer.writeheader()
        safe_row = {key: row.get(key, "") for key in HISTORY_HEADER}
        writer.writerow(safe_row)


def read_history():
    ensure_history_schema()
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{key: row.get(key, "") for key in HISTORY_HEADER} for row in rows]


def write_history(rows):
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_HEADER)
        writer.writeheader()
        for row in rows:
            safe_row = {key: row.get(key, "") for key in HISTORY_HEADER}
            writer.writerow(safe_row)

def delete_file_from_static(url_path):
    if not url_path:
        return
    clean = url_path.lstrip("/").replace("/", os.sep)
    abs_path = os.path.join(BASE_DIR, clean)
    if os.path.exists(abs_path) and os.path.isfile(abs_path):
        os.remove(abs_path)


def create_report(data):
    report_id = data["id"]
    report_path = os.path.join(REPORT_FOLDER, f"laporan_{report_id}.pdf")

    img = Image.new("RGB", (900, 1250), "#ffffff")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        heading_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except:
        title_font = heading_font = text_font = small_font = None

    y = 45
    draw.rounded_rectangle((35, 28, 865, 150), radius=28, fill="#eef6ff")
    draw.text((60, y), "Laporan Hasil Pemeriksaan - CataScan", fill="#0b5fc6", font=title_font)
    y += 65
    draw.text((60, y), f"Tanggal: {data['tanggal']}", fill="#344054", font=text_font)

    y = 190
    draw.text((50, y), f"Status: {data['status']}", fill="#101828", font=heading_font)
    y += 42
    draw.text((50, y), f"Tingkat Risiko: {data['risk_level']}", fill="#101828", font=text_font)
    y += 32
    draw.text((50, y), f"Confidence Katarak: {data['cataract_confidence']}%", fill="#101828", font=text_font)
    y += 32
    draw.text((50, y), f"Confidence Normal: {data['normal_confidence']}%", fill="#101828", font=text_font)
    y += 32
    draw.text((50, y), f"Jumlah Objek Terdeteksi: {data['jumlah_objek']}", fill="#101828", font=text_font)
    y += 32
    draw.text((50, y), f"Kualitas Foto: {data['quality_status']}", fill="#101828", font=text_font)
    y += 55

    draw.text((50, y), "Saran:", fill="#0b5fc6", font=heading_font)
    y += 35
    for line in data["advice_lines"]:
        draw.text((70, y), f"- {line}", fill="#344054", font=text_font)
        y += 30

    y += 25
    draw.text((50, y), "Gambar Hasil Deteksi:", fill="#0b5fc6", font=heading_font)
    y += 40

    result_abs = os.path.join(BASE_DIR, data["result_image"].lstrip("/").replace("/", os.sep))
    if os.path.exists(result_abs):
        result_img = Image.open(result_abs).convert("RGB")
        result_img.thumbnail((800, 520))
        img.paste(result_img, (50, y))
        y += result_img.height + 30

    disclaimer = (
        "Catatan: Hasil ini merupakan prediksi AI untuk screening awal dan bukan diagnosis medis. "
        "Untuk kepastian, lakukan pemeriksaan langsung ke dokter spesialis mata."
    )
    draw.text((50, 1160), disclaimer, fill="#667085", font=small_font)

    img.save(report_path, "PDF", resolution=100.0)
    return f"/static/reports/laporan_{report_id}.pdf"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "CataScan",
        "model_exists": os.path.exists(MODEL_PATH),
        "model_names": model.names
    })


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_path": MODEL_PATH,
        "model_names": model.names,
        "note": "Pastikan class katarak mengandung kata cataract/katarak dan class normal mengandung kata normal/sehat."
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format tidak didukung. Gunakan JPG, JPEG, PNG, atau WEBP."}), 400

    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit(".", 1)[1].lower()
    check_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    unique_name = f"{check_id}.{extension}"

    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    result_path = os.path.join(RESULT_FOLDER, f"result_{unique_name}")
    file.save(upload_path)

    image_quality = analyze_image_quality(upload_path)

    results = model.predict(source=upload_path, conf=0.25)
    result = results[0]

    plotted = result.plot()
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    Image.fromarray(plotted_rgb).save(result_path)

    detections = []

    if result.obb is not None and len(result.obb) > 0:
        for box in result.obb:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100
            class_name = result.names[class_id]
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "is_normal": is_normal_class(class_name),
                "is_cataract": is_cataract_class(class_name)
            })

    decision = classify_detection_result(detections)

    uploaded_image = f"/static/uploads/{unique_name}"
    result_image = f"/static/results/result_{unique_name}"

    action_plan = get_action_plan(decision["status"], decision["risk_level"])
    advice = get_advice(decision["status"], decision["risk_level"])

    response_data = {
        "id": check_id,
        "tanggal": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "status": decision["status"],
        "summary_label": decision["summary_label"],
        "risk_level": decision["risk_level"],
        "risk_class": decision["risk_class"],
        "confidence": decision["confidence"],
        "cataract_confidence": decision["cataract_confidence"],
        "normal_confidence": decision["normal_confidence"],
        "jumlah_objek": len(detections),
        "detections": detections,
        "main_detections": decision["main_detections"],
        "decision_reason": decision["decision_reason"],
        "advice": advice,
        "action_plan": action_plan,
        "image_quality": image_quality,
        "uploaded_image": uploaded_image,
        "result_image": result_image,
        "model_names": result.names,
        "disclaimer": "Hasil ini merupakan prediksi AI dan bukan diagnosis medis. Untuk kepastian, lakukan pemeriksaan langsung ke dokter spesialis mata."
    }

    response_data["voice_text"] = make_voice_text(
        response_data["status"],
        response_data["risk_level"],
        response_data["cataract_confidence"],
        response_data["normal_confidence"],
        response_data["advice"]
    )

    report_url = create_report({
        **response_data,
        "quality_status": image_quality["quality_status"],
        "advice_lines": [advice, decision["decision_reason"]] + action_plan
    })
    response_data["report_url"] = report_url

    save_history({
        "id": check_id,
        "tanggal": response_data["tanggal"],
        "status": response_data["status"],
        "risk_level": response_data["risk_level"],
        "confidence": response_data["confidence"],
        "cataract_confidence": response_data["cataract_confidence"],
        "normal_confidence": response_data["normal_confidence"],
        "jumlah_objek": len(detections),
        "quality_status": image_quality["quality_status"],
        "uploaded_image": uploaded_image,
        "result_image": result_image,
        "report_url": report_url
    })

    return jsonify(response_data)


@app.route("/history")
def history():
    return jsonify(read_history()[::-1])


@app.route("/history/<history_id>", methods=["DELETE"])
def delete_history_item(history_id):
    rows = read_history()
    target = None
    remaining = []

    for row in rows:
        if row.get("id") == history_id:
            target = row
        else:
            remaining.append(row)

    if target is None:
        return jsonify({"error": "Riwayat tidak ditemukan."}), 404

    delete_file_from_static(target.get("uploaded_image"))
    delete_file_from_static(target.get("result_image"))
    delete_file_from_static(target.get("report_url"))

    write_history(remaining)

    return jsonify({"message": "Riwayat berhasil dihapus."})


@app.route("/history", methods=["DELETE"])
def clear_history():
    rows = read_history()

    for row in rows:
        delete_file_from_static(row.get("uploaded_image"))
        delete_file_from_static(row.get("result_image"))
        delete_file_from_static(row.get("report_url"))

    if os.path.exists(HISTORY_CSV):
        os.remove(HISTORY_CSV)

    return jsonify({"message": "Semua riwayat berhasil dihapus."})


@app.route("/dashboard")
def dashboard():
    rows = read_history()
    total = len(rows)
    high = sum(1 for r in rows if r.get("risk_level") == "Risiko Tinggi")
    medium = sum(1 for r in rows if r.get("risk_level") == "Risiko Sedang")
    low = sum(1 for r in rows if r.get("risk_level") == "Risiko Rendah")
    detected = sum(1 for r in rows if r.get("status") in ["TERINDIKASI KATARAK", "BERHASIL TERDETEKSI"])

    confidences = []
    for r in rows:
        try:
            confidences.append(float(r.get("cataract_confidence", 0)))
        except:
            pass

    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0

    return jsonify({
        "total": total,
        "detected": detected,
        "high": high,
        "medium": medium,
        "low": low,
        "avg_confidence": avg_confidence
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
