# Cara Deploy CataScan ke Hugging Face Spaces + Docker

## 1. Buat Space baru

1. Buka Hugging Face.
2. Login.
3. Klik **New Space**.
4. Isi:
   - Space name: `catascan`
   - SDK: **Docker**
   - Visibility: Public atau Private
5. Klik **Create Space**.

## 2. Upload file project

Upload semua isi folder ini ke Space:

```text
app.py
Dockerfile
requirements.txt
README.md
templates/
static/
models/
data/
```

Pastikan model ada di:

```text
models/best_katarak_obb.pt
```

## 3. Tunggu build

Hugging Face akan otomatis build Docker image. Jika berhasil, aplikasi akan muncul di link Space kamu.

## 4. Cek aplikasi

Setelah build selesai, buka:

```text
https://huggingface.co/spaces/USERNAME/catascan
```

Cek status:

```text
https://USERNAME-catascan.hf.space/health
```

Cek class model:

```text
https://USERNAME-catascan.hf.space/model-info
```

## 5. Jika build error

Cek tab **Logs** di Space.

Masalah yang sering terjadi:

- Model `.pt` tidak ada di folder `models/`
- Dependency terlalu berat
- Build timeout
- File terlalu besar

## Catatan kamera

Fitur kamera di browser butuh HTTPS. Link Hugging Face Spaces sudah HTTPS, jadi kamera HP/Android lebih mungkin berjalan dibanding localhost biasa.
