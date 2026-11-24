# Segmentasi instance indoor dan transfer ke foto ponsel

**Studi komputasional · Visi Komputer · NRP 8 · 24 November 2026**

## Abstrak

Studi ini membandingkan YOLOv8n-seg, YOLO11n-seg, dan Mask R-CNN ResNet-50 FPN untuk segmentasi instance pada Panoptic Indoor Segmentation v5. Tiga model awal dilatih pada kondisi seluruh kelas dan tiga kelas sulit yang dipilih lewat validasi. Replikasi 20 epoch pada tiga seed memberi AP validasi 14,65% untuk Mask R-CNN v2 dan 12,03% untuk v1. Pada 359 foto publik, Mask AP checkpoint v2 terpilih mencapai 7,01%, dibanding 4,51% untuk baseline Mask R-CNN. Pada 37 kelas bersama, v2 mencapai 7,15% di test publik dan 7,23% pada 40 foto ponsel. Ketidaklengkapan sebagian anotasi membatasi interpretasi evaluasi eksternal.

## Pertanyaan penelitian

1. Bagaimana peringkat model one-stage dan two-stage pada evaluasi mask dengan grid, pemisahan data, dan metrik yang sama?
2. Bagaimana perubahan dari pelatihan seluruh kelas ke pengkhususan `knife`, `light`, dan `spoon` memengaruhi AP pada ketiga kelas?
3. Apakah keuntungan validasi Mask R-CNN v2 konsisten antar-seed, dan bagaimana hasil model berubah pada foto ponsel lokal?

## Data dan metode

**Data publik.** Pembagian resmi v5 berisi 397 gambar train, 166 validasi, dan 359 test. Sebanyak 74 kelas memiliki minimal satu mask di salah satu split. Hanya 65 kelas punya contoh positif di train. Empat citra diuji dengan overlay untuk mengoreksi orientasi. Proporsi citra dipertahankan pada kanvas 640 × 640 dengan padding 114. Mask dirasterisasi lalu luas dan kotak anotasi dihitung ulang pada grid yang sama. [Dataset v5](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5) · [Logbook](LOGBOOK.md) · [Skrip preprocessing](prepare_dataset.py)

**Foto ponsel.** Koleksi berisi 41 foto. Satu foto selasar yang menghadap halaman terbuka dikeluarkan untuk menjaga cakupan indoor. Dua belas dari 40 foto lainnya ditetapkan sebagai kondisi sulit lewat pemeriksaan visual sebelum analisis prediksi. Mask diekspor dari job Roboflow dan dipetakan ke nama kelas asli. Foto ponsel berada di split evaluasi lokal tanpa peran dalam training atau pemilihan checkpoint. Pada grid 640 px, seluruh JPEG foto ponsel memiliki hash unik dan tidak ada yang identik dengan JPEG split publik. [Catatan audit](TEST_PRIBADI.md) · [Pemeriksaan duplikat](output/test-pribadi/eval-v3/duplikat-byte.json)

**Eksperimen NRP 8.** Tiga pilot menggunakan validasi untuk memilih kelas sulit dari taksonomi resmi. Kelas harus muncul di minimal lima gambar train dan tiga gambar validasi. Untuk beberapa kelas dengan AP pilot yang sama, peringkat menggunakan median luas mask relatif lalu dukungan train. Ketiga arsitektur menghasilkan `knife`, `light`, dan `spoon`. Pasangan model penuh dan tiga kelas dilatih 16 epoch dengan seed 42. Kondisi tiga kelas mengubah ruang kelas dan jumlah supervisi positif. Hanya 43 dari 397 gambar train memuat salah satu target. [Seleksi kelas](hard-classes.json) · [Logbook](LOGBOOK.md)

**Tuning preprocessing.** Enam konfigurasi YOLOv8n-seg memakai 32 epoch, 640 sampel per epoch dan seed 42. Pengulangan contoh langka, resolusi target mask, dan augmentasi HSV diperiksa pada validasi publik. Skor test dilaporkan untuk konfigurasi terpilih setelah pemilihan. [Hasil tuning](output/tuning/HASIL_TUNING.md)

**Replikasi.** Mask R-CNN v1 dan v2 memakai checkpoint COCO bawaan masing-masing. Setiap varian diulang dengan seed 42, 97, dan 123 pada Tesla T4. Batch 2, 20 epoch, AdamW dan cosine annealing, split yang sama, serta evaluasi checkpoint epoch 20 berlaku untuk keenam run. Perbedaan v1 dan v2 mencakup arsitektur dan bobot awal. Varian untuk uji lanjutan dipilih berdasarkan rata-rata AP validasi. Checkpoint representatif dipilih dari seed dengan AP median di varian yang terpilih. [Mask R-CNN, He dkk.](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html) · [Torchvision Mask R-CNN v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) · [Protokol](PROTOKOL_RISET.md)

**Metrik.** AP adalah mask average precision COCO dengan ambang IoU 0,50–0,95, dihitung dari prediksi yang disimpan. AP50 dan AP75 memakai ambang tunggal. Perbandingan antar-domain memakai kelas yang memiliki mask pada kedua set. COCO memberi skor nol bila suatu kelas beranotasi tanpa prediksi yang benar. Kelas tanpa anotasi tidak masuk rata-rata. Batas inferensi 100 deteksi per citra dan ambang confidence 0,005 dipakai dalam evaluasi ini. Inferensi foto ponsel berjalan pada CPU. [Evaluator](metrics.py)

## Hasil pada data publik

| Model | AP seluruh kelas | AP target, pelatihan seluruh kelas | AP target, pelatihan tiga kelas |
|---|---:|---:|---:|
| YOLOv8n-seg | 2,26% | 0,30% | 0,00% |
| YOLO11n-seg | 1,71% | 0,17% | 0,00% |
| Mask R-CNN R50-FPN | **4,51%** | **1,15%** | **0,56%** |

Semua kolom berasal dari 359 gambar test resmi dan memakai mask AP50–95. Pada test publik, pengkhususan tiga kelas menurunkan AP target ketiga model. Mask R-CNN punya AP tertinggi pada dua kondisi pelatihan. Karena jumlah supervisi ikut berubah, eksperimen ini tidak dapat mengisolasi efek ukuran ruang klasifikasi.

Dengan repeat factor, mask ratio 2 menaikkan AP validasi dari 8,32% ke 9,98% pada seed yang sama. Konfigurasi itu memberi AP test semua kelas 2,15%, sementara kontrol seragam memberi 2,22% dan baseline awal memberi 2,26%. Keuntungan validasi belum berarti model lebih baik pada set uji yang terpisah. [Hasil per konfigurasi](output/tuning/HASIL_TUNING.md)

Mask R-CNN memperoleh AP75 kursi 6,58%, dibanding 6,03% untuk YOLOv8n dan 4,61% untuk YOLO11n. Ini mendukung peringkat pada seluruh kursi test publik. Set publik belum memiliki penandaan oklusi yang memungkinkan pengujian langsung untuk kursi tertutup. YOLO11n memperoleh AP botol 5,20% dan AP orang 4,31%. YOLOv8n memperoleh AP botol 5,19% dan AP orang 6,82%. Hasil agregat ini tidak mendukung hipotesis bahwa YOLO11n unggul pada kedua kelas. [Hipotesis sebelum training](LOGBOOK.md) · [Skor per kelas](output/test-pribadi/eval-v3/per-class.csv)

## Replikasi dua varian Mask R-CNN

| Varian | Seed 42 | Seed 97 | Seed 123 | Rata-rata ± SD | Rata-rata menit T4 |
|---|---:|---:|---:|---:|---:|
| R50-FPN v1 | 11,76% | 12,67% | 11,65% | 12,03% ± 0,56 poin | 19,32 |
| R50-FPN v2 | 15,39% | 14,45% | 14,12% | **14,65% ± 0,66 poin** | 22,37 |

V2 unggul pada ketiga pasangan seed dengan selisih rata-rata 2,62 poin persentase pada validasi. Angka simpangan baku merangkum tiga seed. Checkpoint representatif v2 adalah seed 97 dengan AP validasi 14,45%. Sesi v1 seed 42 berakhir ketika menyalin hasil. `summary.json` dan `results.csv` utuh dan lolos verifikasi CRC ZIP. Checkpoint lengkap untuk run itu tidak tersedia. [Tabel run](output/research/validasi-enam-run.csv) · [Kurva training dan sebaran AP](output/research/replikasi-maskrcnn.png)

Evaluasi checkpoint v2 terpilih selesai pada Colab 09 dengan GPU T4. Mask AP pada semua 69 kelas yang memiliki anotasi di test publik meningkat dari 4,51% pada Mask R-CNN baseline ke 7,01% pada v2. Pada tiga target NRP 8, AP meningkat dari 1,15% menjadi 1,57%. Perbandingan test ini memakai checkpoint baseline 16 epoch dan v2 20 epoch. Replikasi dengan jumlah epoch yang sama membandingkan kedua varian pada validasi. [Arsip evaluasi publik v2](output/research/maskrcnn-v2-97/test-public-colab.zip)

![Gambar 1. Mask AP validasi dan kurva loss train untuk tiga seed pada masing-masing varian.](output/research/replikasi-maskrcnn.png)

## Evaluasi eksternal pada foto ponsel

| Model | Publik, 37 kelas | Foto HP, 37 kelas | Selisih foto HP − publik | AP75 foto HP, 37 kelas |
|---|---:|---:|---:|---:|
| YOLOv8n-seg | 2,57% | 2,51% | −0,06 poin | 0,77% |
| YOLO11n-seg | 2,07% | 0,98% | −1,08 poin | 0,54% |
| Mask R-CNN R50-FPN | 4,76% | 4,15% | −0,61 poin | 2,25% |
| Mask R-CNN R50-FPN v2, seed 97 | **7,15%** | **7,23%** | +0,08 poin | **6,89%** |

Set publik memiliki 359 foto dan 69 kelas yang dianotasi. Ekspor v3 foto ponsel memiliki 40 foto indoor, 619 mask yang dapat dipakai, dan 39 kelas yang dianotasi. Irisannya 37 kelas. V2 unggul 2,39 poin pada test publik dan 3,08 poin pada foto HP dibanding Mask R-CNN baseline di kelas-kelas itu. Selisih 0,08 poin antara kedua domain untuk v2 bersifat deskriptif. Cakupan anotasi, ukuran sampel, dan perangkat inferensi turut membatasi perbandingan. [Tabel perbandingan](output/test-pribadi/eval-v3/comparison.csv) · [Prediksi foto untuk v2 seed 97](output/test-pribadi/eval-v3/maskrcnn-v2-97.zip)

![Gambar 2. Mask AP50-95 model penuh pada kelas dengan anotasi di kedua set.](output/test-pribadi/eval-v3/perbandingan-domain.png)

| Model | AP target foto, pelatihan semua kelas | AP target foto, pelatihan tiga kelas |
|---|---:|---:|
| YOLOv8n-seg | 0,64% | 0,00% |
| YOLO11n-seg | 0,09% | 0,00% |
| Mask R-CNN R50-FPN | 0,16% | 0,51% |

Pada foto HP, pengkhususan tiga kelas menurunkan AP target dua YOLO dan menaikkan AP Mask R-CNN. Dukungan sangat timpang. Terdapat satu mask `knife`, 105 mask `light`, serta dua mask `spoon`. Hasil ini hanya mendeskripsikan data yang tersedia. [Jumlah foto dan mask per kelas](output/test-pribadi/eval-v3/photo-test-640.json)

Pada 103 mask kursi, AP bertambah dari 6,46% untuk baseline Mask R-CNN menjadi 12,57% untuk v2. Pada 122 mask orang, AP bertambah dari 12,69% menjadi 22,55%. Pada 12 mask botol, AP turun dari 25,37% menjadi 21,28%. Ketiga kelas menunjukkan bahwa peningkatan agregat v2 tidak merata. [Rincian prediksi v2](output/test-pribadi/eval-v3/maskrcnn-v2-97.zip)

Pada pembagian foto HP, 12 gambar sulit dan 28 gambar lainnya memiliki 23 kelas bersama. AP Mask R-CNN baseline sebesar 3,81% pada kelompok sulit dan 4,90% pada kelompok lain. AP YOLOv8n 1,66% dan 2,24%. AP YOLO11n 2,66% dan 1,30%. Arah perubahan berbeda antar-model. Komposisi objek dan kondisi ruangan juga berbeda di antara kelompok. [Rincian per kelompok](output/test-pribadi/eval-v3/comparison.json)

## Lima contoh kegagalan

Angka adalah IoU mask pada instance yang ditunjukkan. Kelas yang disebutkan untuk suatu prediksi berasal dari mask yang disimpan di atas ambang evaluator. Setiap gambar memperlihatkan anotasi di kiri, lalu prediksi YOLOv8n, YOLO11n, dan Mask R-CNN v1. Kelima foto memiliki gambar dan mask yang identik pada snapshot v2 dan v3. [Verifikasi revisi](output/test-pribadi/revisi-v2-v3.json)

| Foto dan kondisi | Pengamatan dan diagnosis |
|---|---|
| [1000865793 · laptop di depan](output/test-pribadi/eval-v2/cases/1000865793-372.jpg) | Ketiga model memberi mask dengan IoU 0,81–0,87 terhadap area layar, dengan label lain. Mask R-CNN menyebutnya `box` pada IoU 0,85. Overlap tinggi dengan kelas berbeda mendukung diagnosis kesalahan klasifikasi pada contoh ini. |
| [1000865800 · kursi kayu berhimpitan](output/test-pribadi/eval-v2/cases/1000865800-548.jpg) | Mask R-CNN memiliki prediksi `chair` dengan IoU 0,55. Dua YOLO memiliki IoU kelas yang sama mendekati nol. Mask dari kelas lain dapat menutupi kursi lebih baik, misalnya `stairs` pada IoU 0,77 untuk Mask R-CNN. Kedekatan instance dan pilihan kelas dapat berkontribusi pada kasus ini. |
| [1000865809 · lampu redup](output/test-pribadi/eval-v2/cases/1000865809-183.jpg) | Ketiga model tidak menemukan mask `light` yang bertumpang-tindih dengan lampu tipis di plafon. Prediksi tembok atau plafon terdekat meluas jauh dari lampu dan punya IoU maksimum 0,07. Region latar yang besar dapat mendominasi bentuk yang lebih kecil. |
| [1000865811 · dua orang di depan jendela](output/test-pribadi/eval-v2/cases/1000865811-106.jpg) | Mask `person` YOLO11n menutupi kedua orang serta bagian latar. IoU terhadap instance kiri 0,36. YOLOv8n dan Mask R-CNN mencapai 0,79 dan 0,76 pada instance yang sama. Ini menunjukkan perluasan mask ke instance tetangga pada satu konfigurasi. |
| [1000865829 · kursi di ruang redup](output/test-pribadi/eval-v2/cases/1000865829-200.jpg) | Mask YOLOv8n mengikuti bentuk kursi pada IoU 0,89 tetapi memprediksi `person`. YOLO11n dan Mask R-CNN memberi kelas `chair` pada IoU 0,90 dan 0,82. Bentuk mask sudah cukup dekat untuk melewati IoU 0,75. |

IoU dengan kelas dikunci menggabungkan informasi klasifikasi dan batas mask. IoU terbaik tanpa mengunci kelas dipakai sebagai diagnosis tambahan pada lima instance di atas. Mask R-CNN memisahkan prediksi kelas dan mask per proposal. Contoh lokal menunjukkan kedua jenis kesalahan dan tidak mengisolasi satu komponen internal sebagai penyebab. [Arsitektur Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)

## Eksperimen kecil pencahayaan

Gamma 0,65 mencerahkan bagian citra tanpa mengubah padding atau checkpoint. Pada empat foto yang sudah ditandai redup, berisi 58 mask, AP Mask R-CNN baseline berubah dari 3,41% menjadi 4,31%. AP `person` naik dari 8,21% menjadi 20,62% pada enam mask. AP `light` turun dari 4,31% menjadi 3,54% pada 14 mask. Respons ini konsisten dengan kepekaan terhadap pencahayaan pada empat foto tersebut dan berbeda menurut kelas. Uji ini bersifat diagnostik. [Prediksi kontrol dan hasil gamma](output/test-pribadi/gamma-065.json)

## Keterbatasan dan keterlacakan

Sejumlah mask lama menggabungkan beberapa instance berdekatan atau hanya melabeli objek dominan. Instance yang terlewat dapat membuat prediksi yang benar dihitung sebagai false positive. Satu atau dua contoh kelas langka memberi estimasi AP yang sangat sensitif terhadap satu prediksi. Foto berasal dari satu koleksi dan sampel beberapa ruangan mungkin berkorelasi. Replikasi tiga seed menilai variasi training pada satu split publik. Studi ini tidak memiliki estimasi ketidakpastian lintas ruangan atau audit anotator kedua.

Checkpoint, kurva training, prediksi COCO, metadata EXIF, dan lembar audit disimpan pada PC. Foto serta checkpoint berada di direktori yang dikecualikan dari Git. Catatan menyertakan konfigurasi dan kegagalan yang memerlukan pemulihan. Draf ini disiapkan untuk penilaian studi. Deklarasi penggunaan Codex dan Smart Select tersedia terpisah. [Deklarasi AI](DEKLARASI_AI.md) · [Logbook](LOGBOOK.md)

## Kesimpulan

Di bawah protokol ini, Mask R-CNN v1 memimpin tiga baseline pada kedua domain. Replikasi v2 meningkatkan AP validasi pada ketiga seed dan checkpoint terpilih memperoleh AP lebih tinggi pada kedua test set untuk kelas bersama. Nilai absolut beberapa persen dan lima contoh visual menunjukkan model masih kerap salah kelas atau menghasilkan batas mask yang kurang tepat. Pengujian dengan gamma memperlihatkan respons yang berbeda menurut kelas. Kesimpulan tentang transfer ke ruangan lain memerlukan anotasi yang lebih lengkap dan sampel lintas lokasi yang independen.

## Bukti audit dan demo

Gambar di bawah memperlihatkan hasil perubahan pada dua foto. Pada baris pertama, satu mask gabungan diganti dengan mask kursi dan permukaan meja terpisah. Pada baris kedua, mask menutupi objek utama yang sebelumnya kosong. Skrip demo sudah diuji pada satu foto dengan ambang 0,15 dan menghasilkan 16 mask. Untuk sesi penilaian, argumen threshold dapat diubah pada foto yang baru diambil. [Foto demo lokal](output/demo/hasil-ujicoba.png) · [Skrip demo](demo_segmentasi.py)

![Gambar 3. Foto asli, overlay lama, dan overlay setelah perbaikan anotasi per objek.](output/test-pribadi/perbaikan-mask.jpg)

## Referensi metode

He, K., Gkioxari, G., Dollár, P., dan Girshick, R. 2017. Mask R-CNN. IEEE International Conference on Computer Vision. [Artikel](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)

Gupta, A., Dollár, P., dan Girshick, R. 2019. LVIS: A Dataset for Large Vocabulary Instance Segmentation. IEEE/CVF Conference on Computer Vision and Pattern Recognition. [Artikel](https://openaccess.thecvf.com/content_CVPR_2019/html/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.html)

Torchvision. Spesifikasi Mask R-CNN ResNet-50 FPN v2. [Dokumentasi model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)

Ultralytics. Opsi training segmentasi dan resolusi target mask. [Dokumentasi konfigurasi](https://docs.ultralytics.com/usage/cfg)
