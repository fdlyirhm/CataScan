const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const loading = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const resultSummary = document.getElementById("resultSummary");
const uploadedPreview = document.getElementById("uploadedPreview");
const resultPreview = document.getElementById("resultPreview");
const detectionDetails = document.getElementById("detectionDetails");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const downloadReport = document.getElementById("downloadReport");
const soundToggle = document.getElementById("soundToggle");
const testVoice = document.getElementById("testVoice");
const hamburger = document.getElementById("hamburger");
const navLinks = document.getElementById("navLinks");
const clearHistoryBtn = document.getElementById("clearHistory");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startCameraBtn = document.getElementById("startCamera");
const switchCameraBtn = document.getElementById("switchCamera");
const stopCameraBtn = document.getElementById("stopCamera");
const capturePhotoBtn = document.getElementById("capturePhoto");
const locationStatus = document.getElementById("locationStatus");

let cameraStream = null;
let currentFacingMode = "environment";

document.addEventListener("DOMContentLoaded", () => {
    loadDashboard();
    loadHistory();
});

hamburger.addEventListener("click", () => {
    navLinks.classList.toggle("open");
});

navLinks.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", () => navLinks.classList.remove("open"));
});

uploadForm.addEventListener("submit", async function(event) {
    event.preventDefault();

    if (!imageInput.files.length) {
        showToast("Pilih gambar terlebih dahulu.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    await sendPrediction(formData);
});

testVoice.addEventListener("click", () => {
    speak("Suara aktif. Setelah analisis selesai, saya akan membacakan hasil sesuai keputusan sistem, normal atau katarak.");
});

clearHistoryBtn.addEventListener("click", async () => {
    const confirmed = confirm("Yakin ingin menghapus semua riwayat pemeriksaan?");
    if (!confirmed) return;

    const response = await fetch("/history", { method: "DELETE" });
    const data = await response.json();

    showToast(data.message || "Riwayat dihapus.");
    loadHistory();
    loadDashboard();
});

startCameraBtn.addEventListener("click", async function() {
    await startCamera();
});

switchCameraBtn.addEventListener("click", async function() {
    currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
    await startCamera();
});

stopCameraBtn.addEventListener("click", function() {
    stopCamera();
    showToast("Kamera dimatikan.");
});

capturePhotoBtn.addEventListener("click", async function() {
    if (!cameraStream) {
        showToast("Buka kamera terlebih dahulu.");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async function(blob) {
        const formData = new FormData();
        formData.append("image", blob, "camera_capture.jpg");
        await sendPrediction(formData);
    }, "image/jpeg", 0.95);
});

async function startCamera() {
    try {
        stopCamera();

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = cameraStream;
        showToast(currentFacingMode === "environment" ? "Kamera belakang aktif." : "Kamera depan aktif.");
    } catch (error) {
        showToast("Kamera tidak dapat dibuka. Pastikan izin kamera diberikan dan gunakan HTTPS/localhost.");
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        video.srcObject = null;
    }
}

async function sendPrediction(formData) {
    loading.classList.remove("hidden");
    resultSection.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            showToast(data.error || "Terjadi kesalahan saat memproses gambar.");
            return;
        }

        showResult(data);
        loadDashboard();
        loadHistory();
        speakResult(data);
    } catch (error) {
        showToast("Gagal menghubungi server aplikasi.");
    } finally {
        loading.classList.add("hidden");
    }
}

function showResult(data) {
    resultSection.classList.remove("hidden");

    resultSummary.className = `result-summary ${data.risk_class}`;
    resultSummary.innerHTML = `
        <h3>${data.summary_label} • ${data.risk_level}</h3>
        <p><b>Status:</b> ${data.status}</p>
        <p><b>Confidence Katarak:</b> ${data.cataract_confidence}%</p>
        <p><b>Confidence Normal:</b> ${data.normal_confidence}%</p>
        <p><b>Confidence yang ditampilkan:</b> ${data.confidence}%</p>
        <p><b>Jumlah box terdeteksi:</b> ${data.jumlah_objek}</p>
        <p><b>Alasan keputusan:</b> ${data.decision_reason}</p>
        <p><b>Teks suara:</b> ${data.voice_text || "-"}</p>
        <p><b>Saran:</b> ${data.advice}</p>
        <hr>
        <p><b>Kualitas Foto:</b> ${data.image_quality.quality_status} (${data.image_quality.quality_score}/100)</p>
        <p>Pencahayaan: ${data.image_quality.brightness} • Ketajaman: ${data.image_quality.sharpness} • Resolusi: ${data.image_quality.resolution}</p>
        <p>${data.image_quality.recommendation}</p>
        <hr>
        <b>Rekomendasi tindakan:</b>
        <ul>${data.action_plan.map(item => `<li>${item}</li>`).join("")}</ul>
        <small>${data.disclaimer}</small>
    `;

    confidenceFill.style.width = `${data.cataract_confidence}%`;
    confidenceText.textContent = `${data.cataract_confidence}% katarak`;

    uploadedPreview.src = data.uploaded_image;
    resultPreview.src = data.result_image;
    downloadReport.href = data.report_url;

    if (data.detections.length > 0) {
        detectionDetails.innerHTML = data.detections.map((item, index) => {
            const tipe = item.is_cataract ? "Katarak" : item.is_normal ? "Normal" : "Class tidak dikenali";
            return `
                <div class="detail-card">
                    <b>Box ${index + 1}</b><br>
                    Class ID: ${item.class_id}<br>
                    Class Name: ${item.class_name}<br>
                    Tipe keputusan: ${tipe}<br>
                    Confidence: ${item.confidence}%
                </div>
            `;
        }).join("");
    } else {
        detectionDetails.innerHTML = `
            <div class="detail-card">
                Tidak ada objek yang terdeteksi. Coba gunakan foto yang lebih jelas atau pencahayaan lebih baik.
            </div>
        `;
    }

    resultSection.scrollIntoView({ behavior: "smooth" });
}

function speak(text) {
    if (!soundToggle.checked) return;

    if (!("speechSynthesis" in window)) {
        showToast("Browser tidak mendukung fitur suara.");
        return;
    }

    window.speechSynthesis.cancel();

    setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "id-ID";
        utterance.rate = 0.92;
        utterance.pitch = 1;

        const voices = window.speechSynthesis.getVoices();
        const indonesianVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith("id"));
        if (indonesianVoice) {
            utterance.voice = indonesianVoice;
        }

        window.speechSynthesis.speak(utterance);
    }, 120);
}

function speakResult(data) {
    // BUG FIX:
    // Suara tidak lagi membuat keputusan sendiri di frontend.
    // Suara memakai voice_text dari backend agar selalu sesuai hasil normal/katarak.
    if (data.voice_text) {
        speak(data.voice_text);
        return;
    }

    if (data.status === "TERINDIKASI KATARAK") {
        speak(`Hasil analisis menunjukkan indikasi katarak dengan confidence katarak ${Math.round(data.cataract_confidence)} persen.`);
    } else if (data.status === "TIDAK TERINDIKASI KATARAK") {
        speak(`Hasil analisis menunjukkan mata normal dengan confidence normal ${Math.round(data.normal_confidence)} persen.`);
    } else {
        speak(data.advice || "Hasil belum dapat dibacakan.");
    }
}

async function loadHistory() {
    const historyList = document.getElementById("historyList");
    const res = await fetch("/history");
    const data = await res.json();

    if (!data.length) {
        historyList.innerHTML = `<div class="history-card"><div><b>Belum ada riwayat pemeriksaan.</b><br><span>Hasil analisis akan muncul di sini setelah kamu menjalankan deteksi.</span></div></div>`;
        return;
    }

    historyList.innerHTML = data.slice(0, 12).map(item => `
        <div class="history-card">
            <div>
                <b>${item.risk_level}</b><br>
                <span>${item.tanggal}</span><br>
                Status: ${item.status}<br>
                Confidence Katarak: ${item.cataract_confidence || 0}%<br>
                Confidence Normal: ${item.normal_confidence || 0}%<br>
                Kualitas Foto: ${item.quality_status}
            </div>
            <div class="history-actions">
                ${item.report_url ? `<a class="btn btn-soft" href="${item.report_url}" target="_blank">PDF</a>` : ""}
                <button class="btn btn-danger" onclick="deleteHistoryItem('${item.id}')">Hapus</button>
            </div>
        </div>
    `).join("");
}

async function deleteHistoryItem(id) {
    const confirmed = confirm("Hapus riwayat ini?");
    if (!confirmed) return;

    const response = await fetch(`/history/${id}`, { method: "DELETE" });
    const data = await response.json();

    showToast(data.message || "Riwayat dihapus.");
    loadHistory();
    loadDashboard();
}

async function loadDashboard() {
    const res = await fetch("/dashboard");
    const data = await res.json();

    document.getElementById("statTotal").textContent = data.total;
    document.getElementById("statDetected").textContent = data.detected;
    document.getElementById("statHigh").textContent = data.high;
    document.getElementById("statAvg").textContent = `${data.avg_confidence}%`;
}

function findNearby(query) {
    locationStatus.textContent = "Meminta izin lokasi...";

    if (!navigator.geolocation) {
        locationStatus.textContent = "Browser tidak mendukung fitur lokasi.";
        const url = `https://www.google.com/maps/search/${encodeURIComponent(query + " terdekat")}`;
        window.open(url, "_blank");
        return;
    }

    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;

            locationStatus.textContent = `Lokasi ditemukan: ${lat.toFixed(5)}, ${lng.toFixed(5)}`;

            const url = `https://www.google.com/maps/search/${encodeURIComponent(query)}/@${lat},${lng},14z`;
            window.open(url, "_blank");
        },
        function() {
            locationStatus.textContent = "Izin lokasi ditolak. Membuka pencarian umum di Google Maps.";
            const url = `https://www.google.com/maps/search/${encodeURIComponent(query + " terdekat")}`;
            window.open(url, "_blank");
        }
    );
}

function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.remove("hidden");

    clearTimeout(window.toastTimeout);
    window.toastTimeout = setTimeout(() => {
        toast.classList.add("hidden");
    }, 3200);
}
