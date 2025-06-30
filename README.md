# Chatter

Semi-automatic bout segmentation from bird song recordings using acoustic features such as Spectral Flux, Energy, and MFCC coefficients.

## Getting Started
To get Nester started on your device, follow the installation steps below and refer to the [Chatter Manual](https://github.com/mrtnzram/Chatter). *(work in progress)*

1) Open terminal/powershell.
2) Set path to directory where you want to clone the directory: `cd filepath`
3) Clone directory: `git clone https://github.com/mrtnzram/Chatter.git`
4) Create conda environment: `conda create -n env_name python==3.10.16 
pip`
5) Activate environment: `conda activate env_name`
6) Install interactive widget dependencies: `conda install -c conda-forge ipympl jupyterlab`
7) If Jupyter is not already installed, install Jupyter: `pip install jupyter`
8) Install the dependencies using: pip install -r requirements.txt
9) Run Jupyter lab: `jupyter lab`
10) Open Chatter.ipynb

## AudioFeatureExtractor Parameters

- **sr**  
  *default: 22050*  
  The target sampling rate for audio loading.  
  All audio files will be resampled to this rate.

- **n_mfcc**  
  *default: 13*  
  Number of Mel-frequency cepstral coefficients (MFCCs) to compute for each frame.

- **hop_length**  
  *default: 512*  
  Number of samples between successive frames for all frame-based calculations.  
  Controls the time resolution of features.

- **frame_length**  
  *default: 2048*  
  Number of samples per analysis frame for energy and spectral calculations.

- **mfcc_threshold**  
  *default: 0.5*  
  The minimum variance required in the MFCCs for a frame to be considered active.  
  Used in refining active regions.

- **energy_threshold_pct**  
  *float, default: 0.02*  
  The minimum RMS energy required for a frame to be considered active,  
  expressed as a fraction of the maximum RMS energy.

- **min_silence**  
  *default: 0.8*  
  Minimum duration of silence (in seconds) required to separate two bouts.  
  Shorter silences will be merged into a single bout.

- **pad**  
  *default: 0.75*  
  Amount of time (in seconds) to pad around detected bouts before the onset and after the offset of each detected bout for exporting as wav files.

- **active_region_threshold_pct**  
  *default: 0.001*  
  The minimum spectral flux (as a fraction of the maximum) required for a frame to be considered active.

- **min_bout_length**  
  *default: 1.0*  
  Minimum duration (in seconds) for a detected bout to be kept.

- **model**  
  *default: None*  
  Optional classifier model for post-processing or classification of bouts.


# Chatter Widget Features & Usage Guide

## Overview

The Chatter interface provides interactive widgets for exploring, editing, and exporting detected song bouts from your dataset. Below is a quick reference for each widget and button.

---

## Widget Features

### Bird Selection
- **Dropdown:**  
  Selects which bird/song to display and edit.

### Parameter Controls
- **MFCC Thresh:**  
  Minimum variance in MFCCs for a frame to be considered active.
- **Energy Thresh:**  
  Minimum RMS energy (as a fraction of max) for a frame to be considered active.
- **Active Region Thresh:**  
  Minimum spectral flux (as a fraction of max) for a frame to be considered active.
- **Min Silence:**  
  Minimum silence (in seconds) to separate two bouts.
- **Min Bout Len:**  
  Minimum duration (in seconds) for a detected bout.
- **Pad:**  
  Padding (in seconds) before and after each detected bout.

### Bout Selection & Editing
- **Bouts (SelectMultiple):**  
  Select one or more bouts for removal or editing.
- **Onset/Offset:**  
  Edit the start and end time (in seconds) for a selected bout.
- **Update Bout:**  
  Save changes to the onset/offset of the selected bout.
- **Add Bout:**  
  Add a new bout using the current onset/offset values.
- **Remove Bouts:**  
  Remove the selected bouts from the list.

### Actions
- **Finalize Parameters:**  
  Apply and save the current parameter settings for the selected bird/song and update the df dataframe with the new audio feature values. (You will not need to press this button if you are keeping the parameters default)
- **Export Bouts:**  
  Save all detected bouts as separate audio files and appends bout metadata to chatter.bouts_df

---

## How to Use

1. **Select a Bird:**  
   Use the dropdown to choose a bird/song.

2. **Adjust Parameters:**  
   Modify detection parameters as needed. The plot and detected bouts will update automatically. (The default parameters should work well for most use cases)

3. **Edit Bouts:**  
   - Select a bout to edit its onset/offset, then click **Update Bout**.
   - To add a new bout, set onset/offset and click **Add Bout**.
   - To remove bouts, select one or multiple and click **Remove Bouts**.

4. **Finalize & Export:**  
   - Click **Finalize Parameters** to save settings for the current bird.
   - Click **Export Bouts** to save all bouts and append it to the chatter.bouts_df and export the audio clips.
---

# DataFrames in Chatter

## 1. `df` (Main DataFrame)

This is the **primary DataFrame** created by your pipeline (typically via `create_initial_dataset`).  
Each row represents a single audio recording (song) for a bird.

**Typical columns:**
- `species`: Bird species name or code.
- `bird_id`: Unique identifier for the bird.
- `wav_location`: Path to the audio file.
- `song_id`: Unique identifier for the song/recording.
- `audio`: Loaded audio waveform (numpy array).
- `sr`: Sampling rate of the audio.
- `bouts`: List of detected bouts for this song (each bout is a dict with onset/offset/etc.).
- Feature columns (may include):  
  - `rms_energy`: RMS energy per frame.
  - `spectral_flux`: Spectral flux per frame.
  - `mfcc`: MFCCs per frame.
  - `active_regions`, `refined_regions`: Boolean masks for activity/bout detection.
  - ...and any other features computed by your extractor.

---

## 2. `chatter.bouts_df` (Exported Bouts DataFrame)

This DataFrame is created when you export bouts (e.g., with the "Export Bouts" button).  
**Each row represents a single detected bout** (segment of interest) from a song.

**Columns:**
- `species`: Bird species name or code.
- `bird_id`: Unique identifier for the bird.
- `wav_location`: Path to the original audio file.
- `song_id`: Unique identifier for the song/recording.
- `bout_id`: Index of the bout within the song.
- `duration`: Duration of the bout (seconds).
- `onset`: Start time of the bout (seconds).
- `offset`: End time of the bout (seconds).
- `wavstart`: Start time of the exported audio clip (with padding, seconds).
- `wavend`: End time of the exported audio clip (with padding, seconds).
- `intersong_interval`: Time since the previous bout ended (seconds, or `None` for the first bout).
- `bout_wav`: Path to the exported audio file for this bout.

---


## Notes

- I recommend using 30 to 1 minute song recordings, anything longer makes processing take longer when updating bouts and/or selecting new bird recordings.
- The spectrogram and overlays update automatically with parameter changes.
- All edits are reflected in the current session and can be exported at any time.
