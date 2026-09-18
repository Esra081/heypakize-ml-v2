import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import random

# --- CONFIGURATION ---
SR = 16000
DURATION = 1.5
TARGET_SAMPLES = 1000

BASE = Path(__file__).parent.parent
CLEAN_DIR = BASE / "dataset" / "cleaned"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_acoustic_features(y, sr, label):
    """Zaman hizalamasına ihtiyaç duymadan 'Z' ve 'E' harflerindeki tiz zirveleri yakalar."""
    if len(y) < int(sr * DURATION):
        y = np.pad(y, (0, int(sr * DURATION) - len(y)))
    else:
        y = y[:int(sr * DURATION)]

    # 1. Temel Ses Karakteristiği
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    
    # 2. 'Z' ve 'S' gibi sürtünmeli harfleri yakalayan özellikler!
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # ORTALAMA ve MAKSİMUM değerleri alıyoruz. 
    # 'Z' harfi nerede olursa olsun, 'max' değeri sayesinde yakalanacak.
    features = np.concatenate([
        np.mean(mfcc, 1), np.std(mfcc, 1), np.max(mfcc, 1), # Max eklendi
        np.mean(delta, 1), np.std(delta, 1),
        [
            np.mean(centroid), np.max(centroid), # Sesin tiz zirvesi nerede?
            np.mean(rolloff), np.max(rolloff),   # 'Z' harfinin frekans sınırı
            np.mean(rms), np.max(rms)
        ],
        np.mean(contrast, 1)
    ])
    
    return np.append(features, label)

def augment_audio(y, sr, num_required):
    variants = [y]
    while len(variants) < num_required:
        choice = random.choice(['noise', 'fast', 'slow', 'pitch_up', 'pitch_down'])
        if choice == 'noise': variants.append(y + np.random.normal(0, 0.005, len(y)))
        elif choice == 'fast': variants.append(librosa.effects.time_stretch(y, rate=random.uniform(1.05, 1.15)))
        elif choice == 'slow': variants.append(librosa.effects.time_stretch(y, rate=random.uniform(0.85, 0.95)))
        elif choice == 'pitch_up': variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(1, 2)))
        elif choice == 'pitch_down': variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(-2, -1)))
    return variants

def run_feature_pipeline():
    print("🚀 Akustik Zirve Avcısı (Max-Pooling) Başladı...")
    dataset = []
    
    for category, label in [("Hey_Pakize_Positives", 1), ("Not_Hey_Pakize", 0)]:
        files = list((CLEAN_DIR / category).glob("*.wav"))
        if not files: continue
        
        aug_multiplier = max(1, TARGET_SAMPLES // len(files))
        print(f"📦 {category} İşleniyor... (Hedef: ~{len(files) * aug_multiplier} veri)")

        for f in files:
            try:
                y, sr = librosa.load(f, sr=SR)
                variants = augment_audio(y, sr, num_required=aug_multiplier)
                
                for v in variants:
                    dataset.append(get_acoustic_features(v, sr, label))
            except Exception as e:
                print(f"⚠️ Hata: {f.name} -> {e}")

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_DIR / "extracted_features.csv", index=False)
    print(f"\n✅ Tamamlandı! Toplam {df.shape[1]-1} öznitelik çıkarıldı. (Zirve yakalama aktif)")

if __name__ == "__main__":
    run_feature_pipeline()