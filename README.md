# Chatter

Semi-automatic bout segmentation from bird song recordings using acoustic features such as Spectral Flux, Energy, and MFCC coefficients.

## Getting Started
To get Nester started on your device, follow the installation steps below.

1) Open terminal/powershell.
2) Set path to directory where you want to clone the directory: `cd filepath`
3) Clone directory: `git clone https://github.com/mrtnzram/Chatter.git`
4) Create conda environment: `conda create -n env_name python==3.10.16 
pip`
5) Activate environment: `conda activate env_name`
6) Install interactive widget dependencies: `conda install -c conda-forge ipympl jupyterlab`
7) If Jupyter is not already installed, install Jupyter: `pip install jupyter`
8) Install the dependencies using: `pip install -r requirements.txt`
9) Run Jupyter lab: `jupyter lab`
10) Open Chatter.ipynb

## The Process 

Chatter mainly uses librosa's audio extracting features paired with computational techniques and an optional machine learning stage (using your own model or [BirdNet](https://birdnet.cornell.edu/) to detect bouts. This process for detecting bouts in audio signals begins with detecting active regions using spectral flux. Spectral flux is the measure of rate of change in the power spectrum between successive frames. Peaks in spectral flux often correspond to potential active regions. Next, MFCC coefficients pattern recognition is applied to refine these regions further. Mel-Frequency Cepstral Coefficients (MFCCs), which represent the short-term power spectrum of the audio, are extracted and analyzed. When a pattern of atleast three frames is repeated twice, Chatter detects this as a potential active region aswell These identified active regions are then refined by combining RMS energy and MFCC coefficients. This step involves applying thresholds to discard regions with low RMS energy or inconsistent MFCC patterns that may be the product of noise. The result of this stage is a set of potential bouts, which are segments likely to contain the target signals. An optional machine learning model classifier can be used to further validate and classify the detected bouts. Following this, using the interactive widgets, users are able to refine the onset and offsets of these detected bouts, aswell as remove errors and add undetected bouts.

![Chatterprocess](https://github.com/mrtnzram/Chatter/blob/master/Chatterprocess.png)

---
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
- **Mark as Not Outlier**
  Unmark false outlier flags.
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

### Plot
- The X axis is Time(s) incremeted and marked every second with minor increments for every 0.1 seconds to assist in pinpointing bout start and end times
---

## How to Use

1. **Specify Recording Directory**
   In the second cell in the notebook, specify where your recordings are to create the initial dataframe. The script follows the naming convention: `Genus-species-birdid.wav` for extracting the respective bird metadata.

3. **Select a Bird:**  
   Use the dropdown to choose a bird/song.

4. **Adjust Parameters:**  
   Modify detection parameters as needed. The plot and detected bouts will update automatically. (The default parameters should work well for most use cases)

5. **Edit Bouts:**  
   - Select a bout to edit its onset/offset, then click **Update Bout**.
   - If a bout is marked as an outlier even tho it is not, then click **Mark as Not Outlier**
   - To add a new bout, set onset/offset and click **Add Bout**.
   - To remove bouts, select one or multiple and click **Remove Bouts**.

6. **Finalize & Export:**  
   - Click **Finalize Parameters** to save settings for the current bird.
   - Click **Export Bouts** to save all bouts and append it to the chatter.bouts_df and bouts.csv and export the audio clips.
   - Note: if you have an existing bouts.csv **Export Bouts** will recognize this file as the base for chatter.bouts_df and append the file accordingly. This way you can comeback to your progress anytime.
---

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

  *default: 0.02*  

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
 
  Optional classifier model for post-processing or classification of bouts. Takes your models path. If you have an existing model that detects outliers (human speech) among the detected bouts, this may be useful.

- **use_birdnet**

  *default: False*

  Optional clasifier model [BirdNEt](https://birdnet.cornell.edu/) which utilizes bird recognition to detect outlier bouts

- **birdnet_model_path**

  *default: None*

  When use_birdnet = True, make sure to direct the program to where Birdnet is located on your device.

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

- I recommend using 30 second to 1 minute song recordings, anything longer makes processing take longer when updating bouts and/or selecting new bird recordings.
- The spectrogram and overlays update automatically with parameter changes.
- All edits are reflected in the current session and can be exported at any time.
