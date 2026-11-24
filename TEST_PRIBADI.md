# Test set foto pribadi

Folder asli memiliki 41 foto bertanggal pengambilan kamera. Metadata EXIF dan SHA-256 tiap berkas tersedia di [metadata lokal](output/test-pribadi/metadata-exif.json). Foto 1000865815 menghadap area terbuka dari selasar dan dikeluarkan dari evaluasi indoor.

## Kondisi sulit yang terlihat

| Foto | Kondisi | Bagian yang terlihat |
|---|---|---|
| 1000865793.jpg | Oklusi | Laptop di depan menutup bagian tubuh dan kursi. |
| 1000865799.jpg | Oklusi | Orang, kursi, meja, dan laptop bertumpuk pada garis pandang. |
| 1000865800.jpg | Oklusi | Sandaran kursi kayu saling menutupi benda dan orang. |
| 1000865808.jpg | Sudut ekstrem | Laptop dipotret miring dari jarak dekat dengan bagian tepi terpotong. |
| 1000865809.jpg | Cahaya redup | Orang dan kursi tampak gelap di depan proyektor. |
| 1000865810.jpg | Backlight | Jendela terang berada di belakang laptop dan memantul pada layar. |
| 1000865811.jpg | Backlight | Orang dan kursi terlihat sebagai siluet di depan jendela. |
| 1000865813.jpg | Cahaya redup | Papan dan proyeksi lebih terang daripada kursi kayu. |
| 1000865816.jpg | Sudut ekstrem | Rak dipotret miring dengan perspektif tajam dari keranjang di depan. |
| 1000865823.jpg | Oklusi | Sandaran kursi besar menutup meja, kursi lain, dan sebagian ruangan. |
| 1000865828.jpg | Cahaya redup | Meja serta kursi gelap dibanding layar proyeksi. |
| 1000865829.jpg | Cahaya redup | Monitor dan meja gelap dengan latar proyektor terang. |

## Audit ekspor

| Versi ekspor job | Foto dengan anotasi | Entri anotasi | Foto indoor yang dievaluasi | Mask indoor pada 640 px |
|---|---:|---:|---:|---:|
| 24 November, v1 | 40 dari 41 | 607 | 39 | 602 |
| 24 November, v2 | 41 dari 41 | 618 | 40 | 613 |
| 24 November, v3 | 41 dari 41 | 623 | 40 | 619 |

Versi v2 menambahkan 11 mask pada 1000865817. Ada dua `person`, tiga `chair`, tiga `light`, dua `table-merged`, dan satu `backpack`. Pemeriksaan menggunakan overlay asli dan mask, termasuk validasi koordinat, dimensi, nama kelas, poligon, RLE, dan keberadaan anotasi. Mask `clock` pada 1000865831 berada dalam bentuk kotak pada v2, sehingga evaluator membuang entri itu saat persiapan. [Audit v2 per foto](output/test-pribadi/audit-v2/audit.json) · [Seleksi indoor](output/test-pribadi/eval-v2/selection.json)

Versi v3 mengubah kotak jam menjadi mask. Pada 1000865795, tiga kursi diberi mask sendiri dan satu mask meja yang turut menutupi kursi diganti dengan tiga mask permukaan meja. Pemeriksa format COCO tidak menandai masalah pada 41 foto setelah revisi. Dua foto berubah antara v2 dan v3, sementara piksel 40 foto indoor pada grid evaluasi identik. [Overlay sebelum dan sesudah](output/test-pribadi/perbaikan-mask.jpg) · [Perubahan mask](output/test-pribadi/revisi-v2-v3.json) · [Audit v3 per foto](output/test-pribadi/audit-v3/audit.json)

Pada evaluasi v3, `knife` mempunyai satu mask pada satu foto, `light` 105 mask pada 17 foto, dan `spoon` dua mask pada dua foto. AP untuk pisau dan sendok sangat dipengaruhi satu prediksi atau satu anotasi. [Manifest dan jumlah per kelas](output/test-pribadi/eval-v3/photo-test-640.json)

## Temuan peninjauan visual

| Foto | Temuan |
|---|---|
| 1000865795 | Tiga kursi dan tiga permukaan meja sudah dipisah pada v3. |
| 1000865797 | Kursi terlihat pada beberapa posisi dengan hanya satu anotasi kursi. |
| 1000865803 dan 1000865805 | Satu mask kursi menggabungkan bagian dari beberapa kursi. |
| 1000865806 | Sejumlah orang dan kursi berdekatan kurang terpisah pada overlay. |
| 1000865813 | Bentuk beberapa kursi saling menyentuh. Tepi mask di ruang redup perlu diperiksa penuh. |
| 1000865817 | Mask baru mencakup objek utama. Orang dan perabot kecil di latar masih sebagian. |
| 1000865818 | Sendok, wadah makanan, dan makanan sudah ada mask. Mask kursi lama mencakup beberapa kursi. |
| 1000865831 | Kotak jam pada v2 sudah dikonversi menjadi mask pada v3. |

Skor v3 merupakan evaluasi eksploratif. Cakupan objek dan pemisahan instance pada foto yang ditandai masih membatasi interpretasi. Berkas gambar asli dan overlay tidak dimasukkan ke riwayat Git.
