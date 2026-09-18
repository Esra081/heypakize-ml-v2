# 🎙️ HeyPakize — Wake Word Detection System

"Hey Pakize" sesli uyandırma kelimesi (wake word) tespit sistemi. Tıpkı "Hey Siri", "Hey Google" veya "Hey Toyota" gibi çalışır — mikrofonu dinler ve "Hey Pakize" denildiğinde sistemi uyandırır.

## 🚀 Nasıl Çalışır?

1. **Veri Temizleme** (`data_cleaning.py`): Ses kayıtları temizlenir ve standart formata getirilir
2. **Özellik Çıkarma** (`feature_ext.py`): MFCC ve diğer ses özellikleri çıkarılır
3. **Model Eğitimi** (`model_training.py`): Wake word sınıflandırma modeli eğitilir
4. **Canlı Algılama** (`live.py`): Mikrofonu gerçek zamanlı dinler, "Hey Pakize" kelimesini tanır

## 🎯 Proje Amacı

Özel bir wake word sistemi oluşturmak:
- "Hey Pakize" → sistem uyandırılır ✅
- Diğer sesler → sistem uyumaya devam eder ❌

## 🛠️ Kurulum

```bash
pip install -r requirements.txt

# Sırasıyla çalıştır:
python scripts/data_cleaning.py
python scripts/feature_ext.py
python scripts/model_training.py

# Gerçek zamanlı dinleme:
python scripts/live.py
```

## 🔧 Teknolojiler

- Python
- Librosa / PyAudio (ses işleme)
- Scikit-learn / TensorFlow (sınıflandırma)
- MFCC (Mel-Frequency Cepstral Coefficients)
