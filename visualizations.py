import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from IPython.display import display, HTML
import io
import base64

def show_scrollable_figure(fig, min_width=1000):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_data = buf.read()
    img_base64 = base64.b64encode(img_data).decode('ascii')
    html = f"""
    <div style="overflow-x: auto; width: 100%; border: 1px solid #ddd;">
        <img src="data:image/png;base64,{img_base64}" style="min-width: {min_width}px; max-width: none; display: block;">
    </div>
    """
    display(HTML(html))

def plot_spectral_flux_and_active_regions(row):
    spectral_flux = row['spectral_flux']
    active_regions = row['active_regions']
    species = row.get('species', '')
    bird_id = row.get('bird_id', '')

    plt.figure(figsize=(10, 4))
    plt.plot(spectral_flux, color='purple', label="Spectral Flux")
    plt.plot(active_regions * np.max(spectral_flux), color='red', alpha=0.6, label="Active Regions")
    plt.title(f"Active Regions Based on Spectral Flux ({species} {bird_id})")
    plt.xlabel("Frames")
    plt.ylabel("Flux")
    plt.legend(loc='upper right')
    plt.show()

def plot_rms_energy(row):
    rms_energy = row['rms_energy']
    species = row.get('species', '')
    bird_id = row.get('bird_id', '')

    plt.figure(figsize=(10, 4))
    plt.plot(rms_energy, color='orange', label="RMS Energy")
    plt.title(f"RMS Energy Over Time ({species} {bird_id})")
    plt.xlabel("Frames")
    plt.ylabel("Energy")
    plt.legend(loc='upper right')
    plt.show()

def plot_activity_and_mfcc(row, extractor):
    spectral_flux = row['spectral_flux']
    active_regions = row['active_regions']
    rms_energy = row['rms_energy']
    refined_regions = row['refined_regions']
    mfcc = row['mfcc']
    species = row.get('species', '')
    bird_id = row.get('bird_id', '')

    # Align all features
    min_len = min(len(spectral_flux), len(active_regions), len(rms_energy), len(refined_regions))
    spectral_flux = spectral_flux[:min_len]
    active_regions = active_regions[:min_len]
    rms_energy = rms_energy[:min_len]
    refined_regions = refined_regions[:min_len]
    times = np.arange(min_len) * extractor.hop_length / row['sr']

    fig, axs = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1]})

    # Top plot: Activity features (x-axis is time in seconds)
    axs[0].plot(times, spectral_flux, alpha=0.7, color='purple', label="Spectral Flux")
    axs[0].plot(times, active_regions * np.max(spectral_flux), color='red', alpha=0.3, label="Active Regions")
    scaled_rms_energy = rms_energy / np.max(rms_energy) * np.max(spectral_flux)
    axs[0].plot(times, scaled_rms_energy, color='orange', alpha=0.7, label="RMS Energy (Scaled)")
    axs[0].plot(times, refined_regions * np.max(spectral_flux), alpha=0.8, color='blue', label="Refined Active Regions")
    axs[0].set_title(f"Refined Active Regions (MFCC + RMS Energy) ({species} {bird_id})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Activity")
    axs[0].legend(loc='upper right')

    # Bottom plot: MFCC spectrogram (x-axis is time in seconds)
    img = librosa.display.specshow(
        mfcc, x_axis='time', sr=row['sr'], hop_length=extractor.hop_length, ax=axs[1]
    )
    axs[1].set_title("MFCC Spectrogram")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("MFCC Coefficients")
    fig.colorbar(img, ax=axs[1], format='%+2.0f')

    plt.tight_layout()
    plt.show()

def plot_spectrogram_with_bouts(row):
    audio = row['audio']
    sr = row['sr']
    bouts = row['bouts']
    outlier_flags = row.get('outlier_flags', np.zeros(len(bouts), dtype=int))
    species = row.get('species', '')
    bird_id = row.get('bird_id', '')

    duration = len(audio) / sr
    width_inch = max(duration / (5/2), 15)  # Ensure a minimum width to avoid squishing

    fig, ax = plt.subplots(figsize=(width_inch, 4))
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)

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

    ax.set_xticks(np.arange(0, duration + 1, 1))
    ax.set_xticklabels([f"{x:.0f}" for x in np.arange(0, duration + 1, 1)])
    plt.title(f"Spectrogram with Detected Bouts ({species} {bird_id})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_data = buf.read()
    img_base64 = base64.b64encode(img_data).decode('ascii')

    # Create scrollable HTML container
    html = f"""
    <div style="overflow-x: auto; width: 100%; border: 1px solid #ddd;">
        <img src="data:image/png;base64,{img_base64}" style="min-width: 1000px; max-width: none; display: block;">
    </div>
    """
    display(HTML(html))

def plot_spectrogram_base_from_row(row, show_scroll=True):
    audio = row['audio']
    sr = row['sr']
    species = row.get('species', '')
    bird_id = row.get('bird_id', '')
    duration = len(audio) / sr
    width_inch = max(duration / (5/7), 15)  # Ensure a minimum width to avoid squishing

    fig, ax = plt.subplots(figsize=(width_inch, 4))
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)

    # Major ticks every 1s with labels
    major_ticks = np.arange(0, duration + 1, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{x:.0f}" for x in major_ticks])

    # Minor ticks every 0.5s, no labels
    minor_ticks = np.arange(0, duration + 0.1, 0.1)
    ax.set_xticks(minor_ticks, minor=True)

    plt.title(f"Spectrogram with Detected Bouts ({species} {bird_id})",loc = 'left')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()

    if show_scroll:
        show_scrollable_figure(fig)
    return fig, ax, S_db, duration

def plot_bout_overlays(ax, bouts, outlier_flags, S_db, show_scroll=True):
    for i, bout in enumerate(bouts):
        onset = bout['onset']
        offset = bout['offset']
        is_outlier = outlier_flags[i] if len(outlier_flags) == len(bouts) else 0
        color = 'green'
        alpha = 0.3 if is_outlier else 0.15
        ax.axvspan(onset, offset, color=color, alpha=alpha)
        ax.axvline(onset, color=color, linestyle='--')
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

    if show_scroll:
        show_scrollable_figure(ax.figure)

