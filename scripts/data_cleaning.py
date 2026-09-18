import librosa
import numpy as np
import pandas as pd
import shutil
import soundfile as sf
from pathlib import Path

# --- ACADEMIC STANDARDS ---
TARGET_SR = 16000    # Required 16kHz
TARGET_CHANNELS = 1  # Required Mono
MIN_DURATION = 0.5   # Minimum 0.5 seconds

BASE = Path(__file__).parent.parent
RAW_DIR = BASE / "dataset" / "raw"
CLEAN_DIR = BASE / "dataset" / "cleaned"
REPORT_DIR = BASE / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def slugify_filename(text):
    """Converts Turkish characters to English and removes special chars."""
    mapping = str.maketrans("çğıöşüÇĞİÖŞÜ ", "cgiosuCGIOSU_")
    return text.translate(mapping)

def run_data_audit():
    print("🛡️ Pakize v2.1: Data Audit & Filename Normalization Started...")
    audit_results = []

    for category in ["Hey_Pakize_Positives", "Not_Hey_Pakize"]:
        src_path = RAW_DIR / category
        dst_path = CLEAN_DIR / category
        dst_path.mkdir(parents=True, exist_ok=True)
        
        files = list(src_path.glob("*.wav"))
        print(f"\n📂 Processing {category}: {len(files)} files found.")

        for f in files:
            is_valid = False
            fail_reason = "Unknown Error"
            sr, channels, duration = 0, 0, 0
            
            # Normalize filename to English
            clean_name = slugify_filename(f.name)

            try:
                # 1. Physical File Header Check (Fast check)
                info = sf.info(f)
                sr = info.samplerate
                channels = info.channels
                duration = info.duration

                # 2. Requirement Validation
                if duration < MIN_DURATION:
                    fail_reason = f"Short Duration ({duration:.2f}s)"
                elif channels != TARGET_CHANNELS:
                    fail_reason = f"Channel Mismatch ({channels}ch)"
                elif sr != TARGET_SR:
                    fail_reason = f"Sample Rate Mismatch ({sr}Hz)"
                else:
                    # 3. Content Analysis (Silence Detection)
                    y, _ = librosa.load(f, sr=TARGET_SR)
                    if np.max(np.abs(y)) < 0.01:
                        fail_reason = "Silent/Empty Audio"
                    else:
                        is_valid = True
                        fail_reason = "Passed"

            except Exception as e:
                fail_reason = f"Corrupted Format/Read Error"

            audit_results.append({
                "original_name": f.name,
                "clean_name": clean_name if is_valid else "DISCARDED",
                "category": category,
                "sample_rate": sr,
                "channels": channels,
                "duration": round(duration, 2),
                "status": "CLEAN" if is_valid else "DISCARDED",
                "reason": fail_reason
            })

            if is_valid:
                shutil.copy2(f, dst_path / clean_name)
            else:
                print(f"🗑️ DISCARDED: {f.name} -> {fail_reason}")

    # Save English Report
    df = pd.DataFrame(audit_results)
    df.to_csv(REPORT_DIR / "data_audit_report.csv", index=False)
    
    print(f"\n✅ Pipeline Finished!")
    print(f"✨ Cleaned Files: {len(df[df['status']=='CLEAN'])}")
    print(f"🗑️ Discarded Files: {len(df[df['status']=='DISCARDED'])}")
    print(f"📄 Full Audit Log: reports/data_audit_report.csv")

if __name__ == "__main__":
    run_data_audit()