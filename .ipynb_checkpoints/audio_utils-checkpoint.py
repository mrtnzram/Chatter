import os
import numpy as np
import pandas as pd
import librosa
from scipy.ndimage import label
from scipy.signal import butter, filtfilt
from sklearn.metrics.pairwise import cosine_distances
from itertools import groupby
from operator import itemgetter

def create_initial_dataset(root_dir):
    data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                parts = file.replace('.wav', '').split('-')
                if len(parts) >= 2:
                    species = parts[0] + "-" + parts[1]
                    bird_id = parts[2]
                else:
                    species = "unknown"
                    bird_id = "unknown"
                wav_location = os.path.join(root, file)
                data.append({
                    'species': species,
                    'bird_id': bird_id,
                    'wav_location': wav_location,
                    'song_id': 0
                })
    df = pd.DataFrame(data)
    df['song_id'] = df.groupby('species').cumcount()
    return df

def highpass_filter(audio, sr, cutoff=500, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def remove_low_amplitude(audio, threshold_db=-35):
    # Convert to decibels
    rms = np.sqrt(np.mean(audio**2))  # Root Mean Square
    audio_db = 20 * np.log10(np.abs(audio) / rms + 1e-10)  # dB conversion
    # Mask low-decibel values
    mask = audio_db > threshold_db
    return audio * mask

class AudioFeatureExtractor:
    def __init__(
        self,
        sr=22050,
        n_mfcc=13,
        hop_length=512,
        frame_length=2048,
        mfcc_threshold=0.5,
        energy_threshold_pct=0.02,
        min_silence=0.8,
        pad=0.75,
        active_region_threshold_pct=0.05,  # e.g., 15% of max flux
        min_bout_length=1.0,
        model = None
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.mfcc_threshold = mfcc_threshold
        self.energy_threshold_pct = energy_threshold_pct
        self.min_silence = min_silence
        self.pad = pad
        self.active_region_threshold_pct = active_region_threshold_pct
        self.min_bout_length = min_bout_length
        
        self.model = None
        if model:
            self.load_model(model)

    def load_model(self, model_path):
        """Load the model from the given file path."""
        import joblib
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
              
    def classify_bouts(self, bouts, features):
        """Classify each bout as birdsong or not using the model."""
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model first.")

        bout_features = []
        for bout in bouts:
            onset_idx = int(bout['onset'] * self.sr / self.hop_length)
            offset_idx = int(bout['offset'] * self.sr / self.hop_length)
            bout_mfcc = features['mfcc'][:, onset_idx:offset_idx]
            mfcc_mean = bout_mfcc.mean(axis=1)
            mfcc_std = bout_mfcc.std(axis=1)
            feature_vec = np.concatenate([mfcc_mean, mfcc_std])  # Only MFCC features
            bout_features.append(feature_vec)
        preds = self.model.predict(bout_features)
        for bout, pred in zip(bouts, preds):
            bout['is_birdsong'] = pred == 1
        return bouts

    def compute_all_features_and_classify(self, row):
        """Compute all features and classify bouts."""
        features = self.compute_all_features(row)
        bouts = features['bouts']
        if self.model:
            bouts = self.classify_bouts(bouts, features)
        return bouts

    def extract_bout_features(self, audio, sr, bouts):
        features = []
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        for bout in bouts:
            onset_idx = int(bout['onset'] * sr / self.hop_length)
            offset_idx = int(bout['offset'] * sr / self.hop_length)
            bout_mfcc = mfcc[:, onset_idx:offset_idx]
            mfcc_mean = bout_mfcc.mean(axis=1)
            mfcc_std = bout_mfcc.std(axis=1)
            feature_vec = np.concatenate([mfcc_mean, mfcc_std])  # Only MFCC features
            features.append(feature_vec)
        return features

    def flag_outlier_bouts(self, bout_features, threshold=2.0):
        # Compute pairwise distances between all bouts
        if len(bout_features) < 2:
            # Not enough bouts to compare, so no outliers
            return np.zeros(len(bout_features), dtype=int)
        distances = cosine_distances(bout_features)
        avg_dist = distances.mean(axis=1)
        median = np.median(avg_dist)
        mad = np.median(np.abs(avg_dist - median))  # Median Absolute Deviation
        outlier_flags = avg_dist > (median + threshold * mad)
        return outlier_flags.astype(int)  # 1 = outlier, 0 = not

    def load_audio(self, wav_path):
        # Load audio
        audio, sr = librosa.load(wav_path, sr=self.sr)
        # Remove low-decibel values
        audio = remove_low_amplitude(audio, threshold_db=-30)  # Adjust threshold as needed
        # High-pass filter to remove low-frequency noise
        audio = highpass_filter(audio, sr, cutoff=500)  # Adjust cutoff as needed
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio, sr

    def compute_mfcc(self, audio, sr):
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)

    def compute_spectral_flux(self, audio, sr):
        spec = librosa.stft(audio, hop_length=self.hop_length)
        spec_magnitude = np.abs(spec)
        flux = np.sum((np.diff(spec_magnitude, axis=1) ** 2), axis=0)
        return flux
    
    def detect_repeating_mfcc_patterns(self, mfcc, threshold=100, min_length=3, min_repeats=2):
        """
        Detects regions where any MFCC coefficient is above `threshold` for at least `min_length` frames,
        and this pattern repeats at least `min_repeats` times.
        Returns a boolean mask of frames to consider as bouts.
        """
        # mfcc: shape (n_mfcc, n_frames)
        above_thresh = np.any(mfcc > threshold, axis=0)  # shape (n_frames,)
    
        indices = np.where(above_thresh)[0]
        bouts = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            if len(group) >= min_length:
                bouts.append((group[0], group[-1]))
    
        # If the pattern repeats at least min_repeats times, mark those frames as True
        mask = np.zeros(mfcc.shape[1], dtype=bool)
        if len(bouts) >= min_repeats:
            for start, end in bouts:
                mask[start:end+1] = True
        return mask

    def detect_active_regions(self, flux, threshold_pct=None, pad_seconds=0.15):
        if threshold_pct is None:
            threshold_pct = self.active_region_threshold_pct
        threshold = threshold_pct * np.max(flux)
        active = flux > threshold

        # Pad active regions by pad_seconds on both sides
        pad_frames = int(np.round(pad_seconds * 1.0 * self.sr / self.hop_length))
        if pad_frames > 0:
            padded = np.copy(active)
            for i in range(len(active)):
                if active[i]:
                    start = max(0, i - pad_frames)
                    end = min(len(active), i + pad_frames + 1)
                    padded[start:end] = True
            active = padded

        return active

    def compute_rms_energy(self, audio):
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        return rms

    def refine_regions_with_mfcc_and_energy(self, active_regions, mfcc, rms_energy):
        mfcc_variance = mfcc.var(axis=0)
        min_len = min(len(active_regions), len(mfcc_variance), len(rms_energy), mfcc.shape[1])
        active_regions = active_regions[:min_len]
        mfcc_variance = mfcc_variance[:min_len]
        rms_energy = rms_energy[:min_len]
        mfcc1_mask = mfcc[1, :min_len] <= 100  # True where MFCC 1 is <= 100

        # Main criteria (your usual logic)
        energy_threshold = self.energy_threshold_pct * np.max(rms_energy)
        main_criteria_mask = (
            active_regions
            & (mfcc_variance > self.mfcc_threshold)
            & (rms_energy > energy_threshold)
            & mfcc1_mask
        )

        # Additional: repeating MFCC pattern
        pattern_mask = self.detect_repeating_mfcc_patterns(mfcc[:, :min_len], threshold=100, min_length=3, min_repeats=2)

        # Combine with OR: either main criteria OR pattern
        refined_regions = main_criteria_mask | pattern_mask
        return refined_regions

    def get_bouts(self, refined_regions, sr, audio_duration=None):
        labeled, num_segments = label(refined_regions)
        bouts = []
        for i in range(1, num_segments + 1):
            indices = np.where(labeled == i)[0]
            onset = round(indices[0] * self.hop_length / sr,3)
            offset = round(indices[-1] * self.hop_length / sr,3)
            onset_padded = max(0, onset - self.pad)
            if audio_duration is not None:
                offset_padded = min(audio_duration, offset + self.pad)
            else:
                offset_padded = offset + self.pad
            bouts.append({
                'onset': onset,           # unpadded
                'offset': offset,         # unpadded
                'wavstart': onset_padded, # padded
                'wavend': offset_padded   # padded
            })
        if not bouts:
            return []
        merged = [bouts[0].copy()]
        for bout in bouts[1:]:
            prev_offset = merged[-1]['wavend']
            if bout['wavstart'] - prev_offset < self.min_silence:
                merged[-1]['wavend'] = max(merged[-1]['wavend'], bout['wavend'])
                merged[-1]['offset'] = max(merged[-1]['offset'], bout['offset'])
            else:
                merged.append(bout.copy())
        filtered = [b for b in merged if (b['offset'] - b['onset']) >= self.min_bout_length]
        return filtered

    def compute_all_features(self, row):
        audio, sr = self.load_audio(row['wav_location'])
        mfcc = self.compute_mfcc(audio, sr)
        spectral_flux = self.compute_spectral_flux(audio, sr)
        active_regions = self.detect_active_regions(spectral_flux)
        rms_energy = self.compute_rms_energy(audio)
        refined_regions = self.refine_regions_with_mfcc_and_energy(active_regions, mfcc, rms_energy)
        audio_duration = len(audio) / sr
        bouts = self.get_bouts(refined_regions, sr, audio_duration=audio_duration)
        # Compute MFCC features for each bout
        bout_features = self.extract_bout_features(audio, sr, bouts)
        # Flag outliers
        outlier_flags = self.flag_outlier_bouts(np.array(bout_features))
        # Attach outlier flag to each bout
        for bout, flag in zip(bouts, outlier_flags):
            bout['outlier_flag'] = int(flag)
        return pd.Series({
            'audio': audio,
            'sr': sr,
            'mfcc': mfcc,
            'spectral_flux': spectral_flux,
            'active_regions': active_regions,
            'rms_energy': rms_energy,
            'refined_regions': refined_regions,
            'audio_duration': audio_duration,
            'bouts': bouts
        })

def plot_spectrogram_base(S_db, sr, duration, species, bird_id):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(duration / (5/2), 15), 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
    ax.set_xticks(np.arange(0, duration + 1, 1))
    ax.set_xticklabels([f"{x:.0f}" for x in np.arange(0, duration + 1, 1)])
    plt.title(f"Spectrogram ({species} {bird_id})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    return fig, ax

def plot_bout_overlays(ax, bouts, outlier_flags, S_db):
    for i, bout in enumerate(bouts):
        onset = bout['onset']
        offset = bout['offset']
        is_outlier = outlier_flags[i] if len(outlier_flags) == len(bouts) else 0
        color = 'red' if is_outlier else 'green'
        alpha = 0.3 if is_outlier else 0.15
        ax.axvspan(onset, offset, color=color, alpha=alpha)
        ax.axvline(onset, color=color if is_outlier else 'green', linestyle='--')
        ax.axvline(offset, color='blue', linestyle='--')
        ax.text(onset, S_db.shape[0] * 5, f'{onset:.2f}s', color='white', rotation=90, va='bottom', ha='right', fontsize=9)
        ax.text(offset, S_db.shape[0] * 5, f'{offset:.2f}s', color='white', rotation=90, va='bottom', ha='left', fontsize=9)
    for i in range(1, len(bouts)):
        prev_offset = bouts[i - 1]['offset']
        curr_onset = bouts[i]['onset']
        y_bracket = S_db.shape[0] * 0.85
        ax.plot([prev_offset, curr_onset], [y_bracket, y_bracket], color='white', linewidth=2)
        ax.plot([prev_offset, prev_offset], [y_bracket - 5, y_bracket + 5], color='white', linewidth=2)
        ax.plot([curr_onset, curr_onset], [y_bracket - 5, y_bracket + 5], color='white', linewidth=2)
        interval = curr_onset - prev_offset
        ax.text((prev_offset + curr_onset) / 2, y_bracket + 5, f"{interval:.2f}s", color='white', ha='center', va='bottom', fontsize=10)