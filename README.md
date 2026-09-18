# 🤖 HeyPakize — Computer Vision ML Pipeline (v2)

Bilgisayarlı görü alanında geliştirilmiş makine öğrenmesi pipeline projesi. Veri temizleme, özellik çıkarma, model eğitimi ve canlı çıkarım adımlarını içerir.

## 📁 Proje Yapısı

```
scripts/
├── data_cleaning.py      # Ham veriyi temizleme ve ön işleme
├── feature_ext.py        # Özellik çıkarma (feature extraction)
├── model_training.py     # Model eğitimi
└── live.py               # Gerçek zamanlı çıkarım (inference)
```

## 🚀 Kullanım

```bash
pip install -r requirements.txt

# Sırasıyla çalıştır:
python scripts/data_cleaning.py
python scripts/feature_ext.py
python scripts/model_training.py
python scripts/live.py
```

## 🔧 Teknolojiler

- Python, OpenCV
- Scikit-learn / PyTorch
- NumPy, Pandas
