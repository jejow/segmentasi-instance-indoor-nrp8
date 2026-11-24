# Protokol lanjutan untuk uji foto indoor

Bandingkan Mask R-CNN R50-FPN dan R50-FPN v2 menggunakan pretrained COCO masing-masing. Varian v2 merupakan perubahan paket arsitektur dan checkpoint. Enam run memakai seed 42, 97, dan 123 pada masing-masing model, dengan 20 epoch, T4, batch 2, kanvas 640, dan split validasi v5 yang sama. Laporkan mask AP50–95 validasi tiap seed, rata-rata, simpangan baku, dan durasi. Hasil yang mencapai batas waktu sebelum epoch 20 dicatat terpisah.

Aturan pemilihan untuk uji lanjutan memakai varian dengan rata-rata AP validasi tertinggi dan checkpoint dengan AP median dalam varian tersebut. AP test publik dan AP foto pribadi tidak dipakai untuk memilih checkpoint. Aturan ini dicatat sebelum membuka hasil uji checkpoint terpilih.

Foto pribadi dipakai untuk pengujian eksternal. Catat jumlah foto dan mask yang sah per kelas. Laporkan cakupan kelas bersama untuk perbandingan antar-domain. Kelompok cahaya redup, backlight, oklusi, dan sudut ekstrem ditetapkan dari foto sebelum prediksi diperiksa. Analisis lima kandidat kegagalan setelah mencocokkan prediksi dan mask. Interpretasikan skor foto pribadi sebagai eksploratif jika audit masih menemukan instance yang digabung atau terlewat. Data tidak menyediakan identitas ruangan yang terverifikasi untuk bootstrap per ruangan.

Uji kepekaan pencahayaan memakai kontrol dari inferensi yang tersimpan dan gamma 0,65 pada empat foto yang sudah ditetapkan sebagai redup. Ubah luminans input secara monoton per kanal dengan checkpoint yang sama. Gunakan hasilnya untuk diagnosis, tanpa memilih konfigurasi model dari foto ini.

- [Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
- [Torchvision Mask R-CNN R50-FPN v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
