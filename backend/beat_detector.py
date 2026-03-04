import numpy as np
import librosa
import io
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import lightgbm



# web上で呼び出すための設定
app = FastAPI()
@app.get("/")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def extract_features(y, sr, onset_env, bpm_raw, hop_length=512):
    # 前処理(hpssとオンセット強度検出)

    onset_hihat = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmin=4000)
    onset_snare = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmin=2000, fmax=5000)
    onset_bass = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmax=200)

    #オンセット強度の平均
    onset_hihat_mean = np.mean(onset_hihat)
    onset_snare_mean = np.mean(onset_snare)
    onset_bass_mean =np.mean(onset_bass)
    
    # スペクトラルセントロイド抽出
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    centroid_mean = np.mean(spec_centroid)

    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    zcr_mean = np.mean(zcr)

    tg = librosa.autocorrelate(onset_env, max_size=int(4.0 * sr / hop_length))
    

    peaks = librosa.util.peak_pick(tg, pre_max=5, post_max=5, pre_avg=10, post_avg=10, delta=0.05, wait=10)
    peaks_values = tg[peaks]
    sorted_idx = np.argsort(peaks_values)[::-1]
    a1, a2 = 0.0, 0.0
    p1, p2 = 0.0, 0.0

    if len(sorted_idx) >= 1:
        a1 = peaks_values[sorted_idx[0]]
        p1 = peaks[sorted_idx[0]]
    if len(sorted_idx) >= 2:
        a2 = peaks_values[sorted_idx[1]]
        p2 = peaks[sorted_idx[1]]

    a_sum = np.sum(tg) + 1e-6
    
    # onset density        
    duration = len(y) / sr
    onset_density = len(peaks) / duration

    # スペクトラルフラックス
    
    freqs = librosa.tempo_frequencies(len(tg), sr=sr, hop_length=hop_length)

    tau_est_idx = np.argmin(np.abs(freqs - bpm_raw))

    tau_min_idx = 0
    f1 = np.sum(tg[:max(0, tau_est_idx-10)])

    f2 = np.sum(tg[max(0, tau_est_idx -10) : min(len(tg), tau_est_idx + 11)])

    f3 = float(tau_est_idx)

    A1 = a1 / a_sum
    A2 = a2 / a_sum
    A1_A2_ratio = a1 / (a2 + 1e-6)
    P1_P2_ratio = p1 / (p2 + 1e-6)
    Pulse_Clarity = a1 / np.mean(tg)
    ACF_Centroid = np.sum(np.arange(len(tg)) * tg) / a_sum
    ACF_Spread = np.std(tg)

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=384)
    ltp_curve = np.mean(tempogram, axis=1)
    n_dim = 60
    blocks = np.array_split(ltp_curve, n_dim)
    rpv = [np.max(b) if len(b) > 0 else 0 for b in blocks]
    features = [onset_hihat_mean, onset_snare_mean, onset_bass_mean, centroid_mean, zcr_mean, onset_density, f1, f2, f3, A1, A2, A1_A2_ratio, P1_P2_ratio, Pulse_Clarity, ACF_Centroid, ACF_Spread]
    full_features = np.concatenate([features, rpv]) 
    
    
    return full_features

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "octarve_lgb_model.pkl")
lgb_model = joblib.load(model_path)

def bpm_estimate(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
)-> dict:
    
    y_perc = librosa.effects.percussive(y)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr,hop_length=hop_length)
    
    bpm, beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    bpm0 = bpm.item()

    X = extract_features(y_perc, sr, onset_env, bpm0, hop_length=512)
    probs = lgb_model.predict_proba([X])
    predict_label = np.argmax(probs)
    max_probs = np.max(probs, axis=1)
    threshold =0.6
    ratio = {0: 0.5, 1: 0.66, 2: 1.0, 3: 1.5, 4: 2.0}
    if predict_label != 2 and max_probs >= threshold:
        final_ratio = ratio[predict_label]
    else:
        final_ratio = 1.0
        
    final_bpm = bpm0 * final_ratio
    return{
        "bpm_corrected": round(final_bpm)
    }

# if __name__ == "__main__":
#     path = r"C:\Users\tbzbe\Desktop\ファイル\データ分析\beat_detector_jupyter lab\BallroomData\Albums-AnaBelen_Veneo-01.wav"
#     y, sr = librosa.load(path, sr=None, mono=True)
#     result = bpm_estimate(
#         y=y,
#         sr=sr,
#         hop_length=512,

#     )
#     print(result["bpm_corrected"])

    
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    result = bpm_estimate(y, sr, hop_length=512)
    return result["bpm_corrected"]