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
