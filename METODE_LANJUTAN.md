# Perbaikan segmentasi indoor

## Analisis awal

Baseline YOLOv8n-seg dipakai untuk ablasi agar arsitektur, seed 42, bobot awal, batch 8, dan kanvas 640 px sama. Mask R-CNN tetap menjadi pembanding arsitektur dari putaran sebelumnya. Sebanyak 43 dari 397 gambar train memuat satu atau lebih dari knife, light, dan spoon. Kedua YOLO tiga kelas pada baseline mencatat AP mask validasi nol.

## Rancangan dan sumber

- **Pengulangan gambar langka.** Mengadaptasi rumus LVIS yang dipakai Detectron2. Bobot gambar adalah maksimum bobot kelasnya, dengan bobot kelas maksimum antara 1 dan akar dari 0,05 dibagi frekuensi kelas per gambar. Pertahankan 397 gambar asli, lalu undi 243 tambahan sekali dengan seed yang dikunci. Kontrol mengundi jumlah tambahan yang sama secara seragam. Ini adaptasi daftar tetap dengan anggaran yang sama untuk tiap run. [Gupta, Dollar, dan Girshick, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.html) · [Implementasi Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py)
- **Target mask lebih rapat.** Uji mask_ratio 2 vs 4 dengan sampling yang sama. Kanvas gambar tetap 640 px. Perubahan ini memengaruhi rasterisasi target training, sedangkan penilaian akhir memakai evaluator COCO bersama. [Konfigurasi Ultralytics](https://docs.ultralytics.com/usage/cfg)
- **Variasi pencahayaan.** Uji perturbasi saturasi 0,15 dan value 0,25 setelah repeat factor dan mask_ratio 2. Transformasi warna hanya bekerja pada gambar train. [Panduan augmentasi Ultralytics](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- **Tiga kelas NRP 8.** Bandingkan sampling seragam dengan tambahan 243 gambar positif. Semua 397 gambar asli tetap masuk, sehingga varian positif memuat 286 dari 640 sampel dengan foreground. Pakai kelas yang telah dikunci dari pilot awal.

Target 32 epoch dan 640 sampel per epoch untuk enam run. Gunakan checkpoint epoch terakhir dan 166 foto validasi yang sama. Bandingkan hanya dalam kelompok kelas yang sama. Test resmi serta foto HP disisihkan dari pemilihan metode.
