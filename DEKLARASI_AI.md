# Deklarasi penggunaan AI

OpenAI Codex membantu membaca ketentuan NRP 8, merancang hipotesis dan eksperimen, membuat skrip konversi serta training, memvalidasi ekspor, dan mengoperasikan peluncur Colab. Hipotesis adalah draf yang bisa ditinjau dan dipertanggungjawabkan oleh mahasiswa. Angka eksperimen berasal dari evaluasi prediksi terhadap anotasi COCO dan tersimpan dalam artefak run Colab.

## Prompt utama

> lanjudkan tugasnya. kamu train di skill colab pake semua colabnya. kamu kerjakan secara teliti dan efisien untuk colabnya. kerjakan secara teliti

> aku nrp 8

## Contoh saran AI yang perlu diperbaiki

Implementasi awal mencoba menyelaraskan gambar dengan anotasi memakai orientasi EXIF. Uji persiapan menunjukkan empat file potret memiliki dimensi 480×640, sedangkan anotasi merujuk ke 640×480. Keempat file tidak memiliki tag EXIF orientasi. Memaksakan resize akan membuat mask menutupi objek yang keliru.

Perbaikannya adalah menumpuk poligon COCO pada tiga versi tiap gambar dan memeriksa kesesuaian tepi objek. Hasilnya menunjukkan rotasi 90° searah jarum jam untuk tiga foto dan rotasi 90° berlawanan untuk satu foto. Persiapan ulang menghasilkan 11.845 mask yang dapat dirasterisasi pada 922 gambar. Script mencatat setiap koreksi pada manifest data. [Bukti perbandingan overlay](bukti/arah-empat-citra-v5.jpg)

Foto pribadi dikerjakan per objek dengan Smart Select pada bagian yang sudah ditinjau di Roboflow. Bukti proses dan verifikasi akhir anotasi foto pribadi masih perlu diselesaikan sebelum melaporkan AP test eksternal.
