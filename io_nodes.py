import torch
import torchaudio
import os
import glob
import numpy as np

class LoadAudioBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"multiline": False, "default": ""}),
                "file_pattern": ("STRING", {"multiline": False, "default": "*.wav"}),
                "skip_first": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Number of files to skip from the beginning of the sorted list."}),
                "load_cap": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "Maximum number of files to load. -1 means no limit."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_batch",)
    FUNCTION = "load_batch"
    CATEGORY = "L3/AudioTools/IO"

    def load_batch(self, directory_path: str, file_pattern: str, skip_first: int, load_cap: int):
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"ComfyAudio: The specified path is not a valid directory: {directory_path}")

        # Find all files matching the pattern and sort them for predictable order
        search_path = os.path.join(directory_path, file_pattern)
        file_paths = sorted(glob.glob(search_path))

        if not file_paths:
            print(f"ComfyAudio Warning: No files found matching the pattern '{search_path}'. Returning empty audio.")
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100},)

        print(f"ComfyAudio: Found {len(file_paths)} total files.")
        
        # --- APPLY skip_first and load_cap ---
        # 1. Skip the first N files
        if skip_first > 0:
            if skip_first >= len(file_paths):
                print(f"ComfyAudio Warning: 'skip_first' ({skip_first}) is greater than or equal to the number of files found ({len(file_paths)}). Returning empty batch.")
                return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100},)
            file_paths = file_paths[skip_first:]
            print(f"ComfyAudio: Skipped first {skip_first} files, {len(file_paths)} remaining.")

        # 2. Apply the load cap
        if load_cap != -1 and load_cap > 0:
            file_paths = file_paths[:load_cap]
            print(f"ComfyAudio: Capped loading to {len(file_paths)} files.")
        
        if not file_paths:
            print(f"ComfyAudio Warning: No files left to load after applying skip/cap. Returning empty audio.")
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100},)

        waveforms_list = []
        target_sr = None

        # First pass: Load all audio and determine the target format
        for file_path in file_paths:
            try:
                waveform, sr = torchaudio.load(file_path)
                
                if target_sr is None:
                    target_sr = sr
                    
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                    waveform = resampler(waveform)
                
                if waveform.dtype != torch.float32 and torch.max(torch.abs(waveform)) > 1.0:
                    waveform = waveform.to(torch.float32) / torch.iinfo(torch.int16).max
                
                waveforms_list.append(waveform)
            except Exception as e:
                print(f"ComfyAudio Warning: Skipping file '{file_path}' due to an error: {e}")
        
        if not waveforms_list:
            print(f"ComfyAudio Error: All files failed to load. Returning empty audio.")
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100},)
            
        # Second pass: Pad all waveforms to the length of the longest one
        max_len = max(w.shape[1] for w in waveforms_list)
        padded_waveforms = []
        for w in waveforms_list:
            pad_len = max_len - w.shape[1]
            if pad_len > 0:
                padded_w = torch.nn.functional.pad(w, (0, pad_len))
                padded_waveforms.append(padded_w)
            else:
                padded_waveforms.append(w)
        
        batch_tensor = torch.stack(padded_waveforms)
        
        print(f"ComfyAudio: Batch created with shape {batch_tensor.shape} (batch_size, channels, samples)")

        return ({"waveform": batch_tensor, "sample_rate": target_sr},)