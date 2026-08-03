import librosa
import numpy as np

def extract_features(audio_path, sr=16000, duration=3.0):
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        return np.vstack((mel_db, mfcc))
    except Exception as e:
        print(f"Error: {e}")
        return None
