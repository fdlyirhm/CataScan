---
title: CataScan
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CataScan

**Deteksi awal katarak berbasis kecerdasan buatan**

CataScan adalah aplikasi screening awal indikasi katarak yang menggunakan model YOLOv8-OBB untuk menganalisis citra mata, menampilkan hasil deteksi, confidence, tingkat risiko, riwayat, laporan PDF, suara hasil, dan bantuan medis terdekat.

## Fitur

- Upload gambar mata
- Live kamera dari browser
- Putar kamera depan/belakang untuk Android
- Matikan kamera
- Hasil deteksi dengan confidence katarak dan confidence normal
- Risk level
- Cek kualitas foto
- Riwayat pemeriksaan
- Hapus riwayat
- Export laporan PDF
- Suara hasil
- Maps Assist untuk mencari dokter/rumah sakit mata terdekat

## Endpoint penting

- `/` halaman utama
- `/health` cek status aplikasi dan model
- `/model-info` cek nama class model

## Catatan

Aplikasi ini adalah prototype screening awal dan bukan pengganti diagnosis dokter. Untuk kepastian, lakukan pemeriksaan langsung ke dokter spesialis mata.
