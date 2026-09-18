import numpy as np
import librosa
import joblib
import pyaudio
import threading
import customtkinter as ctk
from pathlib import Path
from scipy.signal import butter, filtfilt
from datetime import datetime
import time

# --- CONFIGURATION ---
SR, DURATION, CHUNK = 16000, 1.5, 1024
BASE = Path(__file__).parent.parent
# Eğitimden sonra çıkan modelinin adı neyse buraya onu yazmalısın. 
# (Örn: champion_svm_v5.pkl veya tournament_... içindeki model)
MODEL_PATH = BASE / "outputs" / "champion_svm.pkl" 

class PakizeLoggingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Hey Pakize V5 | Acoustic Peak Hunter Dashboard")
        self.geometry("700x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # --- LOGIC VARIABLES (5'TE 4 HEDEFİ İÇİN KUSURSUZ ORANLAR) ---
        self.dynamic_energy_threshold = 0.015  # Fısıltıları yakalaması için çok hafif hassas
        self.model_threshold = 0.58            # Zirve modelinde en ideal barajdır
        self.hit_counter = 0         
        self.required_hits = 2                 # 2 onay: "Hey Paki"yi eler, "Hey Pakize"yi şak diye yakalar!
        self.peak_confidence = 0.0
        
        self.is_running = True
        self.is_calibrating = False
        
        self.load_models()
        self.setup_ui()
        
        # Ses motorunu arka planda başlat
        threading.Thread(target=self.audio_engine, daemon=True).start()

    def setup_ui(self):
        # Header
        self.header = ctk.CTkLabel(self, text="ACOUSTIC PEAK DASHBOARD", font=("Roboto", 22, "bold"), text_color="#1ABC9C")
        self.header.pack(pady=20)

        # Main Status Indicator
        self.indicator = ctk.CTkFrame(self, width=120, height=120, corner_radius=60, fg_color="#333333", border_width=4, border_color="gray")
        self.indicator.pack(pady=10)
        
        self.status_text = ctk.CTkLabel(self, text="STARTING ENGINE...", font=("Roboto", 18, "italic"), text_color="cyan")
        self.status_text.pack(pady=5)

        # --- GAUGES & BARS ---
        ctk.CTkLabel(self, text="Real-time Voice Energy (RMS):", font=("Roboto", 12)).pack(pady=(15,0))
        self.vocal_bar = ctk.CTkProgressBar(self, width=500, height=10, progress_color="#3498DB")
        self.vocal_bar.set(0)
        self.vocal_bar.pack(pady=5)

        # Confidence Bar with Peak Info
        self.progress_label = ctk.CTkLabel(self, text="Similarity: 0.0% | Peak: 0.0%", font=("Roboto", 16, "bold"))
        self.progress_label.pack(pady=(25, 0))
        self.confidence_bar = ctk.CTkProgressBar(self, width=500, height=20, progress_color="#27AE60")
        self.confidence_bar.set(0)
        self.confidence_bar.pack(pady=10)

        # --- LOGGING AREA ---
        ctk.CTkLabel(self, text="DETECTION HISTORY (LOGS):", font=("Roboto", 13, "bold"), text_color="gray").pack(pady=(20,0))
        self.log_box = ctk.CTkTextbox(self, width=500, height=150, font=("Consolas", 12), fg_color="#1E1E1E")
        self.log_box.pack(pady=10)
        self.log_box.insert("0.0", "--- Pakize V5 Engine Active ---\n")

        # --- CONTROLS ---
        self.btn_recalibrate = ctk.CTkButton(self, text="RE-CALIBRATE ENVIRONMENT", command=self.trigger_calibration, fg_color="#34495E")
        self.btn_recalibrate.pack(pady=10)

        ctk.CTkLabel(self, text="Noise Gate Sensitivity:", font=("Roboto", 11)).pack(pady=(5,0))
        self.sensitivity_slider = ctk.CTkSlider(self, from_=0.001, to=0.05, command=self.update_threshold_manually)
        self.sensitivity_slider.set(self.dynamic_energy_threshold)
        self.sensitivity_slider.pack(pady=(0, 10))

    def add_log(self, confidence):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] DETECTED! Confidence: %{confidence*100:.1f}\n"
        self.log_box.insert("1.0", log_entry)

    def load_models(self):
        try:
            data = joblib.load(MODEL_PATH)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.pca = data["pca"]
            print(f"✅ V5 Akustik Model Yüklendi! (Eğitim Başarısı: %{data['acc']*100:.2f})")
        except Exception as e:
            print(f"❌ Model Yükleme Hatası: Lütfen MODEL_PATH yolunu kontrol et! Hata: {e}")

    def update_threshold_manually(self, value):
        self.dynamic_energy_threshold = value

    def trigger_calibration(self):
        self.is_calibrating = True

    def extract_features_exact(self, audio):
        """Senin attığın V5 kodunun mikrofon için optimize edilmiş BİREBİR aynısı"""
        # Canlı mikrofondaki dip gürültüyü sil (Eğitimde olmayan bir kirlilik bu)
        nyq = SR / 2.0
        b, a = butter(4, 250 / nyq, btype="high")
        audio = filtfilt(b, a, audio)
        
        # Akıllı Normalizasyon (Mikrofon sesi eğitim WAV'larına benzesin diye)
        max_val = np.max(np.abs(audio))
        if max_val > 0.15:  
            audio = audio / max_val

        # Kesin süre (1.5 Saniye)
        if len(audio) < int(SR * DURATION):
            audio = np.pad(audio, (0, int(SR * DURATION) - len(audio)))
        else:
            audio = audio[:int(SR * DURATION)]

        # --- SENİN YAZDIĞIN ÖZNİTELİK SIRALAMASI ---
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        
        centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=SR)
        rms = librosa.feature.rms(y=audio)
        
        feature_vector = np.concatenate([
            np.mean(mfcc, 1), np.std(mfcc, 1), np.max(mfcc, 1), # Max eklendi
            np.mean(delta, 1), np.std(delta, 1),
            [
                np.mean(centroid), np.max(centroid), # Sesin tiz zirvesi nerede?
                np.mean(rolloff), np.max(rolloff),   # 'Z' harfinin frekans sınırı
                np.mean(rms), np.max(rms)
            ],
            np.mean(contrast, 1)
        ])
        
        return feature_vector.reshape(1, -1)

    def ui_safe_update(self, rms, prob, peak, detected):
        """Arayüzü ana thread üzerinden güvenli bir şekilde günceller"""
        self.vocal_bar.set(min(rms * 25, 1.0))
        self.confidence_bar.set(prob)
        self.progress_label.configure(text=f"Similarity: {prob*100:.1f}% | Peak: {peak*100:.1f}%")

        if detected:
            self.status_text.configure(text="🚀 PAKİZE DETECTED!", text_color="#2ECC71")
            self.indicator.configure(fg_color="#2ECC71")
            self.add_log(prob)
        elif not self.is_calibrating:
            self.status_text.configure(text="SYSTEM READY", text_color="cyan")
            self.indicator.configure(fg_color="#333333")

    def audio_engine(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SR, input=True, frames_per_buffer=CHUNK)
        
        buffer = []
        peak_timer = time.time()
        self.status_text.configure(text="SYSTEM READY", text_color="cyan")

        while self.is_running:
            try:
                # 1. Ortam Kalibrasyonu
                if self.is_calibrating:
                    self.status_text.configure(text="ANALYZING NOISE...", text_color="yellow")
                    time.sleep(0.5)
                    temp_frames = []
                    for _ in range(int(SR / CHUNK * 2.0)):
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        temp_frames.append(np.frombuffer(data, dtype=np.float32))
                    
                    noise_level = np.mean([np.sqrt(np.mean(f**2)) for f in temp_frames])
                    self.dynamic_energy_threshold = max(noise_level * 1.5, 0.005)
                    self.sensitivity_slider.set(self.dynamic_energy_threshold)
                    
                    self.is_calibrating = False
                    self.status_text.configure(text="SYSTEM READY", text_color="cyan")
                    continue
                
                # 2. Canlı Ses Okuma
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                current_rms = np.sqrt(np.mean(audio_chunk**2))

                # Sessizlik (Gürültü eşiğinin altı)
                if current_rms < self.dynamic_energy_threshold:
                    self.hit_counter = 0
                    self.after(0, self.ui_safe_update, current_rms, 0.0, self.peak_confidence, False)
                    continue

                # 3. Buffer Doldurma ve Analiz
                buffer.append(audio_chunk)
                if len(buffer) >= int(SR / CHUNK * DURATION):
                    segment = np.concatenate(buffer)
                    buffer.pop(0) # Kayan pencere (Sliding Window)
                    
                    feat = self.extract_features_exact(segment)
                    feat_pca = self.pca.transform(self.scaler.transform(feat))
                    prob = self.model.predict_proba(feat_pca)[0][1]
                    
                    # Peak Takibi
                    if prob > self.peak_confidence:
                        self.peak_confidence = prob
                        peak_timer = time.time()
                    elif time.time() - peak_timer > 3.0: 
                        self.peak_confidence = prob # 3 saniye sonra zirveyi sıfırla

                    # Hit Counter Algoritması (Gürültü duvarı)
                    detected = False
                    if prob >= self.model_threshold:
                        self.hit_counter += 1
                        if self.hit_counter >= self.required_hits:
                            detected = True
                            self.hit_counter = 0
                            buffer = [] # Buffer'ı boşalt ki yankıdan 2. kez tetiklenmesin
                    else:
                        self.hit_counter = 0

                    # UI Güncellemesi
                    self.after(0, self.ui_safe_update, current_rms, prob, self.peak_confidence, detected)
                    
                    if detected:
                        time.sleep(1.2) # Tetiklendikten sonraki bekleme süresi

            except Exception as e:
                print(f"Audio Stream Error: {e}")
                break

if __name__ == "__main__":
    app = PakizeLoggingGUI()
    app.mainloop()