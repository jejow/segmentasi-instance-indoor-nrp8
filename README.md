# Segmentasi instance indoor

Tugas Visi Komputer ITS, CPMK 3, NRP berakhiran 8. YOLOv8n-seg dan YOLO11n-seg mewakili model one-stage, sedangkan Mask R-CNN mewakili two-stage. Setiap model dilatih dalam dua kondisi, semua kelas serta tiga kelas tersulit (`knife`, `light`, dan `spoon`). Pengujian memakai 359 foto test resmi dan 40 foto ponsel.

## Laporan dan bukti

| Berkas | Isi |
|---|---|
| [Laporan](LAPORAN_RISET.md) | Rancangan, hasil pada kedua set, lima studi kasus, dan uji pencahayaan |
| [PDF laporan](laporan-segmentasi-indoor-nrp8.pdf) · [Panel visual](bukti/visual) | Visual data, training, AP per kelas, efek threshold, dan lima studi kasus foto HP dalam bentuk siluet |
| [Lampiran test resmi](bukti/visual/atlas-v5.pdf) · [AP lengkap per kelas](bukti/visual/ap-kelas-lengkap.csv) | Empat adegan v5 dengan acuan COCO dan delapan keluaran model. CSV memuat AP50-95, AP50, AP75, dan jumlah mask |
| [Eksperimen NRP 8](bukti/hasil/nrp8-dua-test.csv) | Kondisi semua kelas dan tiga kelas untuk setiap model. Kolom `*_pct` dalam persen |
| [Skor per kelas](bukti/hasil/ap-per-kelas.csv) | AP dalam skala 0 sampai 1 dan jumlah mask acuan untuk kedua set |
| [Log asli per epoch](bukti/log-training) · [Waktu GPU](bukti/hasil/waktu-gpu.csv) | Salinan `results.csv`, durasi training, seed, dan hash file log |
| [Logbook](LOGBOOK.md) · [Deklarasi AI](DEKLARASI_AI.md) | Hipotesis bertanggal, keputusan, prompt, dan koreksi saran AI |

Angka analisis dan grafik tersedia di [bukti/hasil](bukti/hasil). PDF lokal `output/pdf/laporan-segmentasi-indoor-nrp8.pdf` menyertakan indeks 40 foto HP. Setiap lembar memuat foto masukan, acuan COCO, delapan keluaran model, dan jumlah mask cocok, FP, serta FN. Foto HP, EXIF, dan checkpoint disimpan dalam paket lokal. PDF GitHub menampilkan kelima studi kasus HP sebagai siluet mask.

## Menjalankan eksperimen

1. Ekspor [Panoptic Indoor Segmentation v5](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5) sebagai COCO Segmentation dan simpan sebagai `data/panoptic-indoor-v5-original.coco.zip`. Ekspor resmi punya 397 gambar latih dan 166 gambar validasi.
2. Siapkan kanvas, mask COCO, dan manifest dengan perintah berikut.

   ```powershell
   uv run --no-project --python 3.12 --with pycocotools==2.0.10 --with opencv-python-headless==4.11.0.86 --with pillow python prepare_dataset.py data/panoptic-indoor-v5-original.coco.zip data/panoptic-indoor-v5-640.zip
   ```

3. Siapkan PyTorch dan Torchvision yang sesuai perangkat. Pasang dependensi model melalui `requirements-yolo.txt` atau `requirements-maskrcnn.txt`. Perintah berikut melatih satu kondisi.

   ```powershell
   python train.py --model yolov8n --dataset data/panoptic-indoor-v5-640.zip --seed 42 --epochs 16 --output output/reproduksi/yolov8n-semua.zip
   ```

   Tambahkan `--classes hard-classes.json` untuk kondisi tiga kelas. ID model lain adalah `yolo11n` dan `maskrcnn`.

4. Untuk memeriksa peluncur lokal Colab, jalankan `./run-colab.ps1 -Phase Pilot -Check` dan `./run-colab.ps1 -Phase Full -Check`. Peluncur ini memakai profil Colab dan direktori runtime PC yang digunakan dalam eksperimen.
5. Berikan artefak tiga pilot kepada `select_hard.py` untuk mengulang pemilihan kelas. Gunakan `compare_runs.py` pada artefak run utama untuk membandingkan dua kondisi.

## Evaluasi foto sendiri

Auditi ekspor COCO job Roboflow dengan `audit_private_test.py <ekspor.zip>`. Jalankan `evaluate_private.py <ekspor.zip> <result.zip...> --output <folder>` dalam environment Python yang memiliki dependensi model. Skrip menyiapkan split indoor, mempertahankan taksonomi asli, dan menggunakan checkpoint yang sudah dibekukan. `compare_external.py <folder> <result.zip...> --difficult difficult-photos.json` menghasilkan tabel, skor per kelas, dan grafik kelas yang sama pada kedua domain. [Hasil dan batas evaluasi](LAPORAN_RISET.md)

## Demo satu foto

Jalankan `python demo_segmentasi.py <foto-baru.jpg> --threshold 0.15` dari environment model. Ganti threshold pada argumen untuk memperlihatkan perubahan jumlah mask. Checkpoint bawaan adalah Mask R-CNN v2 seed 97. AP per kelas berasal dari evaluasi berground truth di `per-class.csv`. Foto demo hanya perlu satu gambar baru.

Untuk checkpoint transfer COCO 800 px, gunakan `python demo_segmentasi.py foto-baru.jpg --checkpoint output/improvement/coco-800-97-retry1/checkpoint.pt --rcnn-size 800 --threshold 0.15`. Hasil evaluasi dan pembanding tercatat di [laporan](LAPORAN_RISET.md).

## Membuat ulang grafik dan PDF

```powershell
python -m pip install matplotlib pillow reportlab pypdf
python build_report_evidence.py --figures-only
python render_research_report.py
python render_research_report.py --public --output laporan-segmentasi-indoor-nrp8.pdf
```

Mode `--figures-only` memakai rekap yang tersedia dalam repo. Tiga perintah berikut membuat ulang panel dan grafik tambahan dari arsip evaluasi lokal.

```powershell
uv run --no-project --python 3.12 --with pycocotools==2.0.10 --with pillow --with numpy --with reportlab python build_mask_atlas.py
uv run --no-project --python 3.12 --with pycocotools==2.0.10 --with numpy --with matplotlib python build_visual_metrics.py
uv run --no-project --python 3.12 --with pycocotools==2.0.10 --with pillow --with numpy python build_visual_report.py
```

Lampiran v5 dan HP memakai prediksi tersimpan. Lima studi kasus memakai evaluasi foto HP v2. Lampiran 40 foto memakai v3. [Aturan pemilihan, ambang, dan hash arsip](bukti/visual/ringkasan-overlay.json)
