# 🎓 Student Academic Performance Predictor

Proyek ini bertujuan memprediksi kategori performa akademik mahasiswa (**Rendah / Sedang / Tinggi**) berdasarkan kebiasaan harian seperti jam belajar, screen time (media sosial & Netflix), durasi tidur, kesehatan mental, dan pola hidup lainnya, menggunakan pendekatan *Machine Learning*.

## Latar Belakang

Mahasiswa Generasi Z tumbuh dalam lingkungan yang sangat terhubung dengan teknologi digital, dengan rata-rata screen time yang tinggi setiap harinya. Performa akademik sendiri dipengaruhi oleh kombinasi banyak faktor (jam belajar, screen time, kualitas tidur, kesehatan mental, dll) yang saling berinteraksi secara kompleks dan non-linear — sehingga pendekatan *Machine Learning* relevan digunakan untuk mempelajari pola dari data historis dan memprediksi kategori performa akademik berdasarkan kombinasi kebiasaan tersebut.

## Dataset

Dataset yang digunakan: [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) (Kaggle).

Dataset berisi data kebiasaan harian dan nilai ujian mahasiswa, mencakup fitur seperti jam belajar, screen time, kualitas tidur, kesehatan mental, kehadiran, dan latar belakang pendidikan orang tua.

## Struktur File

| File | Keterangan |
|---|---|
| `notebook.ipynb` | Notebook lengkap: EDA, preprocessing, training, evaluasi, dan penyimpanan model |
| `app.py` | Aplikasi Streamlit untuk demo & inference langsung |
| `student_habits_performance.csv` | Dataset mentah |
| `model_lr.pkl` | Model Logistic Regression (model utama yang dipakai di app) |
| `model_rf.pkl` | Model Random Forest (pembanding) |
| `label_encoders.pkl` | Encoder untuk fitur kategorikal |
| `le_target.pkl` | Encoder untuk label target (kategori performa) |
| `scaler.pkl` | StandardScaler untuk fitur numerik |
| `requirements.txt` | Daftar library yang dibutuhkan |

## Model & Hasil Evaluasi

Dua algoritma dibandingkan pada dataset yang sama:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.745 | 0.749 | 0.745 | 0.746 |
| **Logistic Regression** | **0.800** | **0.805** | **0.800** | **0.802** |

Logistic Regression dipilih sebagai model final karena memberikan performa lebih baik dibandingkan Random Forest, mengindikasikan bahwa relasi antar fitur (terutama `study_hours_per_day`) terhadap kategori nilai bersifat cenderung linear. Kedua model masih mengalami kesulitan membedakan kelas "Sedang" karena posisinya di antara kategori "Rendah" dan "Tinggi" sehingga batasnya kurang tegas.

## Cara Menjalankan Secara Lokal

1. Clone repository ini:
   ```bash
   git clone <link-repo>
   cd <nama-folder-repo>
   ```

2. (Opsional, disarankan) buat virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

4. (Opsional) Jalankan ulang notebook untuk melihat proses training dari awal:
   ```bash
   jupyter notebook notebook.ipynb
   ```
   *Tidak wajib dijalankan ulang karena file model (`.pkl`) sudah disediakan langsung di repo ini.*

5. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

6. Buka browser ke alamat yang muncul di terminal (biasanya `http://localhost:8501`).

## Live Demo

🔗 https://najwanr-performance-predictor.streamlit.app/

## Disclaimer

Prediksi pada aplikasi ini murni berdasarkan model *Machine Learning* yang dilatih menggunakan sampel data tertentu, dan ditujukan sebagai media refleksi diri — bukan diagnosis akademik mutlak.
