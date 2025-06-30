import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, Audio, HTML
import ipywidgets as widgets
from visualizations import *
import librosa

class Chatter:
    def __init__(self, df, extractor):
        self.spectrogram_cache = {}
        self.df = df
        self.extractor = extractor
        for idx, row in self.df.iterrows():
            if 'audio' in row and 'sr' in row and row['audio'] is not None and row['sr'] is not None:
                try:
                    # Use get_cached_spectrogram to populate cache
                    self.get_cached_spectrogram(idx, row['audio'], row['sr'])
                except Exception as e:
                    print(f"Error processing spectrogram for row {idx}: {e}")

        # --- Output Widget ---
        self.plot_output = widgets.Output()

        # --- Style and layout for widgets ---
        self.style = {'description_width': '150px'}
        self.layout_wide = widgets.Layout(width='400px', margin='5px')

        # --- Storage ---
        self.current_bouts = {}

        # --- Dropdown ---
        self.dropdown = widgets.Dropdown(
            options=[(f"{row['species']} {row['bird_id']}", idx) for idx, row in self.df.iterrows()],
            value=0,
            description='Select Bird:'
        )

        # --- Parameter widgets ---
        self.mfcc_threshold = widgets.FloatText(value=0.5, step=0.1, description='MFCC Thresh:', style=self.style, layout=self.layout_wide, format='.2f')
        self.energy_threshold = widgets.FloatText(value=0.1, step=0.01, description='Energy Thresh:', style=self.style, layout=self.layout_wide, format='.2f')
        self.active_region_thresh = widgets.FloatText(value=0.001, step=0.01, description='Active Region Thresh:', style=self.style, layout=self.layout_wide, format='.2f')
        self.min_silence = widgets.FloatText(value=0.9, step=0.1, description='Min Silence:', style=self.style, layout=self.layout_wide, format='.2f')
        self.min_bout_len = widgets.FloatText(value=1.0, step=0.1, description='Min Bout Len:', style=self.style, layout=self.layout_wide, format='.2f')
        self.pad = widgets.FloatText(value=0.5, step=0.1, description='Pad:', style=self.style, layout=self.layout_wide, format='.2f')

        # --- Bout Selection ---
        self.bout_select = widgets.SelectMultiple(
            options=[],
            description='Bouts:',
            style=self.style,
            layout=self.layout_wide
        )

        self.onset_box = widgets.FloatText(
            value=round(0.0, 3), step=0.01, description='Onset:', style=self.style, layout=self.layout_wide, format='.3f', disabled = False
        )
        self.offset_box = widgets.FloatText(
            value=round(0.0, 3), step=0.01, description='Offset:', style=self.style, layout=self.layout_wide, format='.3f', disabled = False
        )
        self.update_bout_btn = widgets.Button(description="Update Bout", button_style='warning')
        self.output_update_bout = widgets.Output()

        # --- Store per-bird parameter state ---
        self.bird_params = {}

        # --- Button and output widgets ---
        self.finalize_btn = widgets.Button(description="Finalize Parameters", button_style='success')
        self.output_finalize = widgets.Output()

        self.save_bouts_btn = widgets.Button(description="Export Bouts", button_style='info')
        self.output_save_bouts = widgets.Output()

        self.remove_bouts_btn = widgets.Button(description="Remove Bouts", button_style='danger', layout=widgets.Layout(width='200px'))
        self.add_bout_btn = widgets.Button(description="Add Bout", button_style='success')  # New Button
        
        self.output_remove_bouts = widgets.Output()
        self.output_add_bout = widgets.Output()

        # --- Setup handlers ---
        self.dropdown.observe(self._on_bird_change)
        self.finalize_btn.on_click(self._on_finalize_clicked)
        self.save_bouts_btn.on_click(self._on_save_bouts_clicked)
        self.remove_bouts_btn.on_click(self._on_remove_bouts_clicked)
        self.bout_select.observe(self._on_bout_select_change, names='value')
        self.update_bout_btn.on_click(self._on_update_bout_clicked)
        self.add_bout_btn.on_click(self.on_add_bout_clicked)

        # For optimized plotting
        self.current_ax = None
        self.current_fig = None
        self.current_S_db = None
        self.current_row_idx = None

        # Initialize params for first bird
        self._load_params_for_bird(self.dropdown.value)

        # Setup interactive output
        self.interactive_output = widgets.interactive_output(
            self.update_plot,
            {
                'idx': self.dropdown,
                'mfcc_threshold_val': self.mfcc_threshold,
                'energy_threshold_val': self.energy_threshold,
                'active_region_thresh_val': self.active_region_thresh,
                'min_silence_val': self.min_silence,
                'min_bout_len_val': self.min_bout_len,
                'pad_val': self.pad
            }
        )

        # Display the first bird by default
        self._draw_base_and_overlay(self.dropdown.value)

    def _draw_base_and_overlay(self, idx):
        row = self.df.iloc[idx].copy()
        # Use cached spectrogram
        S_db, sr = self.get_cached_spectrogram(idx, row['audio'], row['sr'])
        plot_row = row.copy()
        plot_row['audio'] = row['audio']
        plot_row['sr'] = row['sr']
        plot_row['species'] = row.get('species', '')
        plot_row['bird_id'] = row.get('bird_id', '')
        with self.plot_output:
            clear_output(wait=True)
            plt.close('all')
            # Draw base spectrogram (do NOT show or close yet)
            fig, ax, S_db, _ = plot_spectrogram_base_from_row(plot_row, show_scroll=False)
            # Draw overlays on the same axes
            plot_bout_overlays(
                ax,
                plot_row['bouts'],
                np.array([b.get('outlier_flag', 0) for b in plot_row['bouts']]),
                S_db,
                show_scroll=False
            )
            # Now show the scrollable figure with overlays included
            show_scrollable_figure(fig)
            audio_widget = Audio(filename=row['wav_location'])
            html = HTML(f'<div style="text-align:center;">{audio_widget._repr_html_()}</div>')
            display(html)

    def get_cached_spectrogram(self, idx, audio, sr):
        if idx in self.spectrogram_cache:
            return self.spectrogram_cache[idx]
        S = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        self.spectrogram_cache[idx] = (S_db, sr)
        return S_db, sr

    def _load_params_for_bird(self, idx):
        if idx in self.bird_params:
            params = self.bird_params[idx]
        else:
            params = {
                'mfcc_threshold': self.mfcc_threshold.value,
                'energy_threshold': self.energy_threshold.value,
                'active_region_thresh': self.active_region_thresh.value,
                'min_silence': self.min_silence.value,
                'min_bout_len': self.min_bout_len.value,
                'pad': self.pad.value
            }
            self.bird_params[idx] = params
        self.mfcc_threshold.value = params['mfcc_threshold']
        self.energy_threshold.value = params['energy_threshold']
        self.active_region_thresh.value = params['active_region_thresh']
        self.min_silence.value = params['min_silence']
        self.min_bout_len.value = params['min_bout_len']
        self.pad.value = params['pad']

    def _save_params_for_bird(self, idx):
        self.bird_params[idx] = {
            'mfcc_threshold': self.mfcc_threshold.value,
            'energy_threshold': self.energy_threshold.value,
            'active_region_thresh': self.active_region_thresh.value,
            'min_silence': self.min_silence.value,
            'min_bout_len': self.min_bout_len.value,
            'pad': self.pad.value
        }

    def update_plot(self, idx, mfcc_threshold_val, energy_threshold_val, active_region_thresh_val, min_silence_val, min_bout_len_val, pad_val):
        self._save_params_for_bird(idx)
        self.extractor.mfcc_threshold = mfcc_threshold_val
        self.extractor.energy_threshold_pct = energy_threshold_val
        self.extractor.active_region_threshold_pct = active_region_thresh_val
        self.extractor.min_silence = min_silence_val
        self.extractor.min_bout_length = min_bout_len_val
        self.extractor.pad = pad_val

        row = self.df.iloc[idx].copy()
        if 'audio' not in row or 'sr' not in row or row['audio'] is None or row['sr'] is None:
            audio, sr = self.extractor.load_audio(row['wav_location'])
            row['audio'] = audio
            row['sr'] = sr
            self.df.at[idx, 'audio'] = audio
            self.df.at[idx, 'sr'] = sr

        features = self.extractor.compute_all_features(row)
        combined = {**row.to_dict(), **features}

        if 'bouts' in row and isinstance(row['bouts'], list) and len(row['bouts']) > 0:
            combined['bouts'] = row['bouts']

        self.current_bouts[idx] = combined['bouts']

        bout_options = []
        for i, bout in enumerate(combined['bouts']):
            duration = bout['offset'] - bout['onset']
            label = f"Bout {i}: {bout['onset']:.2f}-{bout['offset']:.2f}s | Duration: {duration:.2f}s"
            bout_options.append((label, i))
        self.bout_select.options = bout_options

        with self.plot_output:
            clear_output(wait=True)
            plt.close('all')
            # Only redraw base if new bird/recording
            if self.current_row_idx != idx or self.current_ax is None:
                self._draw_base_and_overlay(idx)
            else:
                # Only update overlays dynamically
                plot_bout_overlays(
                    self.current_ax,
                    combined['bouts'],
                    np.array([b.get('outlier_flag', 0) for b in combined['bouts']]),
                    self.current_S_db,
                    show_scroll=False
                )
                audio_widget = Audio(filename=row['wav_location'])
                html = HTML(f'<div style="text-align:center;">{audio_widget._repr_html_()}</div>')
                display(html)

    def _on_bird_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            # Clear outputs to avoid overlapping
            self.output_finalize.clear_output()
            self.output_save_bouts.clear_output()
    
            # Load parameters for the selected bird
            self._load_params_for_bird(change['new'])
    
            # Update only if the selected bird is different
            if self.current_row_idx != change['new']:
                self._draw_base_and_overlay(change['new'])
                self.current_row_idx = change['new']


    def _on_finalize_clicked(self, b):
        idx = self.dropdown.value
        params = self.bird_params[idx]

        self.extractor.mfcc_threshold = params['mfcc_threshold']
        self.extractor.energy_threshold_pct = params['energy_threshold']
        self.extractor.active_region_threshold_pct = params['active_region_thresh']
        self.extractor.min_silence = params['min_silence']
        self.extractor.min_bout_length = params['min_bout_len']
        self.extractor.pad = params['pad']

        row = self.df.iloc[idx].copy()
        features = self.extractor.compute_all_features(row)
        for key, value in features.items():
            self.df.at[idx, key] = value

        with self.output_finalize:
            clear_output()
            print(f"Parameters and features finalized for {self.df.loc[idx, 'species']} {self.df.loc[idx, 'bird_id']}")

    def _on_save_bouts_clicked(self, b):
        bout_rows = []
        save_dir = "bouts_audio"
        os.makedirs(save_dir, exist_ok=True)
        for idx, row in self.df.iterrows():
            prev_offset = None
            audio, sr = row['audio'], row['sr']
            for bout_id, bout in enumerate(row['bouts']):
                onset = bout['onset']
                offset = bout['offset']
                wavstart = bout['wavstart']
                wavend = bout['wavend']
                if bout_id == 0:
                    intersong = None
                else:
                    intersong = onset - prev_offset
                duration = offset - onset
                bout_audio = audio[int(wavstart * sr):int(wavend * sr)]
                bout_filename = f"{row['species']}_{row['bird_id']}_bout{bout_id}.wav"
                bout_path = os.path.join(save_dir, bout_filename)
                import soundfile as sf
                sf.write(bout_path, bout_audio, sr)
                bout_rows.append({
                    'species': row['species'],
                    'bird_id': row['bird_id'],
                    'wav_location': row['wav_location'],
                    'song_id': row['song_id'],
                    'bout_id': bout_id,
                    'duration': duration,
                    'onset': onset,
                    'offset': offset,
                    'wavstart': wavstart,
                    'wavend': wavend,
                    'intersong_interval': intersong,
                    'bout_wav': bout_path
                })
                prev_offset = offset
        self.bouts_df = pd.DataFrame(bout_rows)
        with self.output_save_bouts:
            clear_output()
            print(f"Audio files saved to {save_dir}")

    def _on_remove_bouts_clicked(self, b):
        idx = self.dropdown.value
        bouts = self.current_bouts.get(idx, [])
        to_remove = set(self.bout_select.value)
        new_bouts = [b for i, b in enumerate(bouts) if i not in to_remove]

        self.df.at[idx, 'bouts'] = new_bouts
        self.current_bouts[idx] = new_bouts

        with self.output_remove_bouts:
            clear_output()
            print(f"Removed bouts: {sorted(to_remove)}")

        self.update_plot(
            idx,
            self.mfcc_threshold.value,
            self.energy_threshold.value,
            self.active_region_thresh.value,
            self.min_silence.value,
            self.min_bout_len.value,
            self.pad.value
        )

    def _on_bout_select_change(self, change):
        idx = self.dropdown.value
        bouts = self.current_bouts.get(idx, [])
        selected = list(self.bout_select.value)
        
        # Update onset and offset only for a single selected bout
        if len(selected) == 1 and selected[0] < len(bouts):
            bout = bouts[selected[0]]
            self.onset_box.value = bout['onset']
            self.offset_box.value = bout['offset']
            self.onset_box.disabled = False
            self.offset_box.disabled = False
            self.update_bout_btn.disabled = False
        else:
            self.onset_box.value = 0.0
            self.offset_box.value = 0.0
            self.onset_box.disabled = False
            self.offset_box.disabled = False
            self.update_bout_btn.disabled = False


    def _on_update_bout_clicked(self, b):
        idx = self.dropdown.value
        selected = list(self.bout_select.value)
        if len(selected) == 1 and selected[0] < len(self.current_bouts[idx]):
            bout_id = selected[0]
            bouts = self.current_bouts[idx]
            new_onset = self.onset_box.value
            new_offset = self.offset_box.value
            bouts[bout_id]['onset'] = new_onset
            bouts[bout_id]['offset'] = new_offset

            pad_val = self.pad.value
            row = self.df.iloc[idx]
            audio_len_sec = len(row['audio']) / row['sr'] if 'audio' in row and 'sr' in row else np.inf
            bouts[bout_id]['wavstart'] = max(new_onset - pad_val, 0)
            bouts[bout_id]['wavend'] = min(new_offset + pad_val, audio_len_sec)

            self.df.at[idx, 'bouts'] = bouts
            self.current_bouts[idx] = bouts

            with self.output_update_bout:
                clear_output()
                print(f"Updated Bout {bout_id}: Onset={new_onset:.2f}, Offset={new_offset:.2f}, wavstart={bouts[bout_id]['wavstart']:.2f}, wavend={bouts[bout_id]['wavend']:.2f}")

            self.update_plot(
                idx,
                self.mfcc_threshold.value,
                self.energy_threshold.value,
                self.active_region_thresh.value,
                self.min_silence.value,
                self.min_bout_len.value,
                self.pad.value
            )

    def on_add_bout_clicked(self, b):
        idx = self.dropdown.value
        bouts = self.current_bouts.get(idx, [])
        new_onset = self.onset_box.value
        new_offset = self.offset_box.value
    
        # Validate that onset is less than offset
        if new_onset >= new_offset:
            with self.output_add_bout:
                clear_output()
                print("Error: Onset must be less than Offset.")
            return
    
        # Add the new bout
        new_bout = {
            'onset': round(new_onset, 3),
            'offset': round(new_offset, 3),
            'wavstart': round(max(0, new_onset - self.pad.value), 3),
            'wavend': round(min(len(self.df.iloc[idx]['audio']) / self.df.iloc[idx]['sr'], new_offset + self.pad.value), 3)
        }
        bouts.append(new_bout)
        bouts.sort(key=lambda b: b['onset'])  # Ensure bouts are sorted
    
        self.df.at[idx, 'bouts'] = bouts
        self.current_bouts[idx] = bouts
    
        with self.output_add_bout:
            clear_output()
            print(f"Added new bout: Onset={new_bout['onset']}, Offset={new_bout['offset']}")
    
        self.update_plot(
            idx,
            self.mfcc_threshold.value,
            self.energy_threshold.value,
            self.active_region_thresh.value,
            self.min_silence.value,
            self.min_bout_len.value,
            self.pad.value,
        )


    def display(self):
        row1 = widgets.HBox([self.dropdown], layout=widgets.Layout(margin='10px'))
        row2 = widgets.HBox([self.mfcc_threshold, self.energy_threshold, self.active_region_thresh], layout=widgets.Layout(margin='10px'))
        row3 = widgets.HBox([self.min_silence, self.min_bout_len, self.pad], layout=widgets.Layout(margin='10px'))
        row4 = widgets.HBox([self.finalize_btn, self.save_bouts_btn], layout=widgets.Layout(margin='10px', justify_content='center', align_items='center'))

        row_bout_select = widgets.HBox([self.bout_select, self.remove_bouts_btn], layout=widgets.Layout(margin='10px'))
        row_bout_edit = widgets.HBox([self.onset_box, self.offset_box, self.update_bout_btn,self.add_bout_btn], layout=widgets.Layout(margin='10px'))

        ui = widgets.VBox([
            row1, row2, row3, row_bout_select, row_bout_edit, row4,
            self.output_update_bout, self.output_remove_bouts, self.output_finalize, self.output_save_bouts,
            self.plot_output
        ], layout=widgets.Layout(margin='20px'))

        display(ui, self.interactive_output)