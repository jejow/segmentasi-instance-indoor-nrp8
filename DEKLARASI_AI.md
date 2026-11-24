# Deklarasi penggunaan AI

OpenAI Codex membantu membaca ketentuan NRP 8, merancang hipotesis dan eksperimen, membuat skrip konversi serta training, memvalidasi ekspor, dan mengoperasikan peluncur Colab. Hipotesis adalah draf yang bisa ditinjau dan dipertanggungjawabkan oleh mahasiswa. Angka eksperimen berasal dari evaluasi prediksi terhadap anotasi COCO dan tersimpan dalam artefak run Colab.

## Prompt utama

> lanjudkan tugasnya. kamu train di skill colab pake semua colabnya. kamu kerjakan secara teliti dan efisien untuk colabnya. kerjakan secara teliti

> aku nrp 8

> kerjakan dengan foto itu. kerjakan denganteliti sampe layak

## Contoh saran AI yang perlu diperbaiki

Implementasi awal mencoba menyelaraskan gambar dengan anotasi memakai orientasi EXIF. Uji persiapan menunjukkan empat file potret memiliki dimensi 480×640, sedangkan anotasi merujuk ke 640×480. Keempat file tidak memiliki tag EXIF orientasi. Memaksakan resize akan membuat mask menutupi objek yang keliru.

Perbaikannya adalah menumpuk poligon COCO pada tiga versi tiap gambar dan memeriksa kesesuaian tepi objek. Hasilnya menunjukkan rotasi 90° searah jarum jam untuk tiga foto dan rotasi 90° berlawanan untuk satu foto. Persiapan ulang menghasilkan 11.845 mask yang dapat dirasterisasi pada 922 gambar. Script mencatat setiap koreksi pada manifest data. [Bukti perbandingan overlay](bukti/arah-empat-citra-v5.jpg)

Codex membantu mengoperasikan Smart Select per objek, mengoreksi kelas serta kontur saat memeriksa overlay, dan menulis skrip evaluasi foto pribadi. Fitur Roboflow Find Objects with AI tidak dipakai dalam pengerjaan ini. Ekspor v3 memiliki mask pada seluruh foto indoor tetapi sebagian objek masih terlewat atau digabung. Status ini dicatat saat melaporkan angka test eksternal eksploratif. Foto ponsel dievaluasi lokal.

Saat pengisian kelas, editor sempat menerima teks lama dari clipboard. Mask meja kemudian diperbaiki menjadi `table-merged` melalui pilihan kelas yang diperiksa di layar. Audit v3 tidak menemukan anotasi yang memakai kelas di luar taksonomi publik.
