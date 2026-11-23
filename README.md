# Segmentasi instance indoor · NRP 8

Eksperimen membandingkan YOLOv8n-seg dan YOLO11n-seg (one-stage) dengan Mask R-CNN ResNet-50 FPN (two-stage). Setiap model memiliki kondisi pelatihan semua kelas dan tiga kelas tersulit. Hipotesis dan keputusan eksperimen tercatat di [logbook](LOGBOOK.md).

Dataset publik [Panoptic Indoor Segmentation v5](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5) menggunakan lisensi CC BY 4.0. Ekspor resmi berisi 397 gambar train, 166 gambar validasi, dan 359 gambar test. Ada 74 kelas dengan mask di setidaknya satu split dan 65 kelas dengan mask di train. Kelas yang absen dari train tetap ada dalam label head kondisi penuh. Mask AP makro COCO menghitung kelas yang punya ground truth di split tersebut.

## Menjalankan di PC ini

1. Ekspor v5 sebagai COCO Segmentation dan simpan ZIP sebagai `data/panoptic-indoor-v5-original.coco.zip`.
2. Siapkan kanvas, mask COCO, dan manifest dengan perintah berikut.

   ```powershell
   uv run --no-project --python 3.12 --with pycocotools==2.0.10 --with opencv-python-headless==4.11.0.86 --with pillow python prepare_dataset.py data/panoptic-indoor-v5-original.coco.zip data/panoptic-indoor-v5-640.zip
   ```

3. Periksa paket dengan `.un-colab.ps1 -Phase Pilot -Check` dan `.un-colab.ps1 -Phase Full -Check`. Perintah tanpa `-Check` memakai profil 01–03 untuk pilot dan 04–06 untuk semua kelas. Gunakan ID berbeda melalui `-RunSuffix` untuk eksperimen ulang.
4. Setelah tiga pilot selesai, berikan tiga artefak `result.zip` kepada `select_hard.py` sesuai posisi argumen. Gunakan manifest hasil analisis untuk `.un-colab.ps1 -Phase Three -Check` dan jalankan fase Three pada profil 07–09.
5. Berikan artefak run utama kepada `compare_runs.py` untuk menghasilkan CSV dan grafik NRP 8. Seluruh tabel AP memakai metrik mask yang sama. Progres dan unduhan hasil ada di [Pantau Colab](http://127.0.0.1:8766/03_colab/?proyek=segmentasi-indoor-nrp8#progres).

## Evaluasi foto sendiri

Siapkan ekspor COCO dari foto yang sudah diaudit di Roboflow sebagai data eksternal. Gunakan `prepare_dataset.py --external-taxonomy data/panoptic-indoor-v5-640.json` untuk mencocokkan nama kelas dan menghasilkan satu split private. Pair evaluasi memakai `train.py --checkpoint <file> --dataset <zip> --eval-splits private`. Sertakan `--classes hard-classes.json` untuk model tiga kelas. Bandingkan kelas yang memiliki ground truth di kedua test set dan laporkan cakupan kelas.

## Artefak

Setiap run utama menghasilkan checkpoint, log per epoch `results.csv`, prediksi COCO, ringkasan AP dan catatan waktu GPU. Fase pilot hanya memakai data train dan validasi untuk menetapkan kelas sulit. File ZIP dataset dan checkpoint disimpan di luar riwayat Git.
