# Logbook segmentasi instance indoor, NRP 8

## 2026-11-24 02:36:32 +07:00 · Rencana dan hipotesis sebelum training

- Hipotesis utama ialah Mask R-CNN ResNet-50 FPN menghasilkan mask AP75 tertinggi untuk kursi yang berdekatan atau tertutup sebagian. ROIAlign dan satu head mask per proposal dapat mempertahankan batas antar-instance. Mask 28×28 berpotensi merugikan benda sangat tipis atau lubang kecil.
- YOLO11n-seg diperkirakan lebih kuat daripada YOLOv8n-seg pada person dan bottle dalam kondisi cukup terang karena perbedaan blok ekstraksi fitur. Keduanya berpotensi kehilangan tepi tipis setelah proyeksi prototype mask. Hipotesis ini diuji dengan AP mask per kelas dan AP75.
- Model tiga kelas diperkirakan memperoleh AP rata-rata pada kelas target yang lebih tinggi karena ruang klasifikasi lebih kecil. Selisih dapat mengecil pada test set pribadi akibat pencahayaan dan oklusi yang berbeda.
- Gunakan Panoptic Indoor Segmentation v5, pembagian resmi 397 train, 166 valid, dan 359 test. Foto HP adalah test set eksternal. Pemilihan kelas, checkpoint, dan hiperparameter memakai train/valid.
- Profil 01–03 menjalankan pilot tiga epoch, masing-masing satu arsitektur. Urutkan kelas berdasarkan rata-rata mask AP50–95 pilot pada validasi. Syarat awal minimal lima gambar train dan tiga gambar valid dengan kelas itu. Jika seri, prioritaskan median luas mask relatif terkecil, lalu jumlah gambar train lebih sedikit. Evaluasi cakupan tiap kelas dalam dua test set setelah pemilihan dikunci.
- Profil 04–06 menjalankan model semua kelas dan 07–09 menjalankan pasangan tiga kelas. Bandingkan setiap pasangan pada gambar test dan tiga kelas yang sama. Catat AP mask makro seluruh kelas untuk model penuh secara terpisah. Seed 42, target 16 epoch, kanvas bersama 640 px, dan batas waktu training per run 45 menit. Evaluasi checkpoint epoch akhir dan catat jika dibatasi waktu.
- Catat tanggal mulai aktual, GPU, waktu, data yang dipakai, kurva training, dan masalah runtime dari log asli.

Sumber dataset: https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5

## 2026-11-24 02:45:01 +07:00 · Validasi ekspor sebelum training

- Endpoint Roboflow mengonfirmasi versi 5 dan pembagian resmi. ZIP COCO Segmentation memuat 922 gambar dengan 4.479 mask train, 2.022 mask validasi, dan 5.344 mask test sebelum normalisasi.
- Pemeriksaan awal menghentikan persiapan karena empat file berukuran 480×640, sementara koordinat COCO merujuk ke 640×480. Saya membandingkan overlay asli, rotasi searah jarum jam, dan rotasi berlawanan. Tiga gambar cocok dengan rotasi searah jarum jam dan satu gambar cocok dengan rotasi berlawanan. Bukti ada di [gambar overlay](bukti/arah-empat-citra-v5.jpg). Solusi awal AI yang hanya mengandalkan EXIF gagal karena keempatnya tidak memiliki tag orientasi. Persiapan kini memperbaiki empat nama file yang telah diverifikasi.
- Evaluasi bersama memakai mask COCO yang diproyeksikan ke kanvas 640×640 dengan rasio aspek asli dipertahankan. Luas dan bbox dihitung ulang dari raster mask. JSON asli tetap tersimpan. Label kelas resmi tidak digabungkan.

## 2026-11-24 02:56:01 +07:00 · Peluncuran paralel

- Paket publik berisi 11.845 mask dan 74 kelas aktif di seluruh split. Ada 65 kelas dengan contoh positif di train. Hash SHA-256 paket 640 px adalah 3bc2e86693d941dc5119e748137053c8b6bc9ce1bf408323ddd1129fb50a5ec4. Pemeriksaan byte gambar tidak menemukan duplikat persis lintas split.
- Monitor beberapa profil gagal mengimpor paket dari environment UV sementara. Saya memasang versi CLI yang sama di virtual environment lokal yang persisten dan melanjutkan dengan ID permintaan runtime yang sudah tercatat. Run 04 dan 05 telah melaporkan batch nyata.
- Diperiksa 41 foto asli dari folder Mobile Devices. Seluruhnya memiliki EXIF dan waktu pengambilan asli. Bukti metadata disimpan secara lokal dan dikecualikan dari Git.

## 2026-11-24 03:05:28 +07:00 · Kelas sulit dan pemulihan evaluasi

- Pilot tiga epoch selesai pada T4, dengan AP mask validasi semua kelas 2,66% (YOLOv8n), 2,24% (YOLO11n), dan 4,99% (Mask R-CNN). Ketiganya memilih **knife, light, spoon** menurut aturan peringkat yang direncanakan. Ketiga kelas memiliki AP pilot 0 dan menempati urutan teratas setelah seri dipecah memakai luas mask median relatif. Detail lengkap ada di peringkat-kesulitan.csv.
- Pada test resmi v5 terdapat 19 mask knife pada 8 gambar, 295 mask light pada 98 gambar, dan 36 mask spoon pada 17 gambar. Jumlah ini diperiksa setelah daftar kelas dikunci.
- Run penuh YOLOv8n dan YOLO11n mencapai 16 epoch, tetapi evaluasi pertama gagal mengemas prediksi. COCO.loadRes mengubah objek prediksi dengan menambahkan area bertipe NumPy uint32 dan bbox array. Evaluator kini memberi COCO salinan objek. Uji prediksi identik menghasilkan AP 1 dan prediksi tetap dapat diserialisasi setelah evaluasi. Riwayat asli kedua percobaan disimpan.
- YOLOv8n mengulang pada profil 04. Pengulangan YOLO11n pada profil 05 ditolak oleh Colab sehingga pengulangan dialihkan ke profil 02 setelah pilot profil itu selesai. Mask R-CNN penuh di profil 06 memiliki salinan checkpoint akhir yang diminta dari runtime lama agar 16 epoch tidak perlu diulang.
