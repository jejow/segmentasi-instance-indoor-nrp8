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

## 2026-11-24 03:32:31 +07:00 · Enam run utama selesai

- Verifikasi akhir menerima enam checkpoint 16 epoch dan tiga arsip pilot dari Pantau Colab. Manifest hasil menyimpan ukuran, profil, dan SHA-256 setiap arsip. Log dan CSV asli tersedia bersama checkpoint di output/runs.
- Pada test resmi, AP mask semua kelas adalah 2,26% untuk YOLOv8n-seg, 1,71% untuk YOLO11n-seg, dan 4,51% untuk Mask R-CNN R50-FPN.
- Untuk knife, light, dan spoon, AP kondisi penuh vs tiga kelas adalah 0,30% dan 0,00% pada YOLOv8n, 0,17% dan 0,00% pada YOLO11n, serta 1,15% dan 0,56% pada Mask R-CNN. Hanya 43 dari 397 gambar train memiliki satu atau lebih kelas target. Pada dua YOLO kondisi tiga kelas, prediksi mask tidak mencapai ambang 0,005.
- Checkpoint Mask R-CNN profil 06 dipulihkan secara utuh dan dievaluasi lewat profil 03. Run tiga kelas profil 09 menyelesaikan semua epoch dengan satu batch tanpa RoI yang memakai supervisi RPN dan satu forward yang diulang dengan FP32. Log kegagalan awal dipertahankan sebagai bukti proses.
- Sumber uji pribadi yang tersedia lokal masih berupa ekspor awal dengan mask pada tiga gambar. AP foto pribadi dan analisis domain shift menunggu ekspor hasil audit.

## 2026-11-24 16:11:12 +07:00 · Evaluasi lanjutan selesai

- Enam run YOLOv8n-seg selesai 32 epoch. Pemenang validasi semua kelas ialah repeat factor + mask ratio 2 dengan AP 9,98%. Kontrol seragam memperoleh 9,65% dan AP tiga target tertinggi 1,56%.
- Setelah pemilihan dari validasi, tiga checkpoint dievaluasi pada 359 foto test resmi. AP semua kelas repeat factor + mask ratio 2 adalah 2,15%, kontrol seragam 2,22%, dan YOLOv8n baseline awal 2,26%. AP tiga target di test adalah 0,51%, 0,35%, dan 0,30%. Mask R-CNN baseline masih tertinggi dengan AP semua kelas 4,51% dan AP tiga target 1,15%.
- Kedua run yang dilatih hanya pada tiga kelas mencatat AP validasi 0%. Tie diputuskan untuk kontrol seragam, yang mendapat AP tiga target 0,23% pada test resmi.
- Arsip model, prediksi, log, tabel, dan grafik terverifikasi dan disimpan di output/tuning.

## 2026-11-24 · Audit foto HP dan replikasi Mask R-CNN

- Ekspor pertama untuk audit memuat 41 foto dan 607 entri anotasi. Satu foto kelas belum berisi mask, meski tercatat selesai di job. Sebelas mask Smart Select disimpan pada foto tersebut. Saya memisahkan tiga kursi dan tiga permukaan meja pada 1000865795 serta mengonversi satu kotak jam pada 1000865831 menjadi mask. Ekspor v3 memuat 623 mask. Setelah mengecualikan foto yang menghadap halaman, set evaluasi memuat 40 foto, 619 mask, dan 12 kondisi sulit yang dipilih sebelum pemeriksaan prediksi.
- Enam run T4 mencapai 20 epoch. Unduhan v1 seed 42 terputus saat penyalinan. Dua entri awal arsip (`summary.json` dan `results.csv`) dapat dibaca utuh dan cocok dengan CRC masing-masing. Lima checkpoint lain tersedia lengkap. Rata-rata AP validasi v1 12,03% dan v2 14,65%. Checkpoint v2 seed 97 dipilih melalui median AP di varian dengan rata-rata tertinggi.
- Pemilihan checkpoint memakai split validasi publik. Pengukuran foto HP memakai checkpoint yang dibekukan dan inferensi CPU lokal. Prediksi dipakai ulang pada revisi v3 hanya setelah byte gambar 640 px cocok, lalu dihitung ulang dengan mask v3. Perbandingan awal menunjukkan 37 kelas bersama untuk model penuh.
- Empat foto redup diuji ulang dengan gamma 0,65 dan checkpoint Mask R-CNN baseline yang sama. Mask AP pada 58 mask berubah dari 3,41% menjadi 4,31%. Uji ini berguna untuk memeriksa sensitivitas cahaya pada sampel kecil.
- Evaluasi v2 seed 97 pada 359 foto publik selesai di Colab 09. Percobaan ulang memakai transfer dalam bagian kecil dan memperbarui koneksi sesi. Arsip test lolos pemeriksaan CRC dan hash split sumber cocok. AP semua 69 kelas yang memiliki anotasi adalah 7,01%. Pada 37 kelas bersama, skor publik dan foto HP adalah 7,15% dan 7,23%.
- Bukti data dan hasil tersedia di direktori output/test-pribadi dan output/research. Ringkasan serta CSV loss asli keenam run disalin ke [log run](output/research/log-runs). Foto, metadata, checkpoint, dan prediksi tidak dimasukkan ke Git.

## 2026-11-24 21:18 WIB · Uji resolusi dan inisialisasi Mask R-CNN v2

- Dari 5.344 mask acuan test publik, 42,2% tidak mempunyai prediksi dengan mask IoU minimal 0,50 dan 17,8% hanya mempunyai prediksi bertumpang tindih dari kelas lain. Sebanyak 14,1% mask kecil berpasangan dengan prediksi sekelas pada ambang itu, dibandingkan dengan 60,4% mask besar.
- Kunci rancangan faktorial dengan seed 42 dan 97 sebelum meluncurkan training. Bandingkan inisialisasi acak dengan transfer prediktor COCO pada 33 kelas yang namanya persis sama. Ukuran internal 640 dan 800 px memakai kanvas acuan 640 px yang sama. Kontrol 640 px tersedia dari replikasi sebelumnya. [Protokol uji](PROTOKOL_PERBAIKAN.md)
- Paket Colab berisi 397 gambar train dan 166 gambar validasi dari split publik yang sama. Pemeriksaan hash anggota arsip mencocokkan kedua split dengan paket v5. Test publik dan foto HP menjadi data untuk pengukuran terpisah dari pemilihan konfigurasi.

## 2026-11-24 21:40 WIB · Koreksi instans pada tiga foto kelas

- Pada 1000865803, ganti satu mask kursi gabungan dengan lima mask kursi yang dapat dibedakan. Potongan kayu di tepi kiri tidak ditetapkan sebagai instans terpisah karena identitasnya ambigu.
- Pada 1000865805, ganti satu mask kursi gabungan dengan tiga mask kursi, termasuk satu kursi terpotong di kanan.
- Pada 1000865797, tambahkan empat mask kursi dan satu mask jam. Setiap objek dipilih dengan satu area Smart Select yang ditinjau di editor, lalu disimpan menggunakan kelas yang sudah ada.
- Revisi terlihat kembali pada job setelah halaman dimuat ulang. Angka penelitian foto HP yang sudah dibukukan merujuk pada ekspor v3. Ekspor baru belum terverifikasi pada pukul 21:40 WIB.

## 2026-11-24 23:12 WIB · Evaluasi checkpoint transfer COCO

- Enam run Colab baru dan dua kontrol yang diarsipkan masing-masing memiliki catatan 20 epoch lengkap. Rata-rata Mask AP validasi pada acak 640, transfer 640, acak 800, dan transfer 800 berturut-turut 14,92%, 15,27%, 13,37%, dan 15,36%. Rancangan yang sudah dikunci memilih transfer 800 dan checkpoint seed 97. AP objek kecil di validasi tertinggi pada transfer 640. [Tabel run](output/improvement/validation.csv)
- Uji publik selesai pada T4 setelah transfer artefak pertama terputus. Percobaan yang berhasil mengirim arsip dalam bagian 3 MiB dan menyegarkan koneksi runtime. Perhitungan ulang prediksi COCO untuk 359 gambar mencocokkan AP model lama 7,01% dan checkpoint terpilih 7,74%. AP tiga kelas NRP 8 berubah dari 1,57% menjadi 2,29%. [Rincian evaluasi publik](output/improvement/public-test.json)
- Inferensi lokal pada 40 foto dan 619 mask yang sama menghasilkan AP 7,23% dan 7,44% pada 37 kelas bersama. AP objek kecil pada foto HP naik dari 1,45% ke 3,51%. Evaluasi ini memakai snapshot v3. Koreksi terbaru di Roboflow tercatat pada entri pukul 21:40 WIB. [Rincian evaluasi foto HP](output/improvement/phone-test.json)
