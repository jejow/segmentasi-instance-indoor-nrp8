# Protokol lanjutan untuk uji foto indoor

Bandingkan Mask R-CNN R50-FPN dan R50-FPN v2 menggunakan pretrained COCO masing-masing. Varian v2 merupakan perubahan paket arsitektur dan checkpoint. Enam run memakai seed 42, 97, dan 123 pada masing-masing model, dengan 20 epoch, T4, batch 2, kanvas 640, dan split validasi v5 yang sama. Laporkan mask AP50–95 validasi tiap seed, rata-rata, simpangan baku, dan durasi. Hasil yang mencapai batas waktu sebelum epoch 20 dicatat terpisah.

Foto pribadi dipakai untuk pengujian eksternal setelah anotasi diverifikasi. Catat jumlah foto dan mask yang sah per kelas. Laporkan cakupan kelas bersama untuk perbandingan antar-domain. Kelompok cahaya redup, backlight, oklusi, dan sudut ekstrem ditetapkan dari foto sebelum prediksi diperiksa. Analisis lima kegagalan setelah evaluasi resmi. Bootstrap sebaiknya mengambil ulang kelompok ruangan atau sesi pengambilan bila pengenal kelompok tersedia, karena foto dari satu ruangan saling berkaitan.

- [Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
- [Torchvision Mask R-CNN R50-FPN v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
