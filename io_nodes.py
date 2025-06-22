import torch
import torchaudio
import os
import glob
import numpy as np

class LoadAudioBatch:
    # Define sort options as a class attribute for clarity
    SORT_ORDER_CHOICES = [
        "Alphabetical (A-Z)",
        "Alphabetical (Z-A)",
        "Date/Time (Latest First)",
        "Date/Time (Oldest First)",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"multiline": False, "default": ""}),
                "file_pattern": ("STRING", {"multiline": False, "default": "*.wav"}),
                "sort_order": (s.SORT_ORDER_CHOICES, {"default": "Alphabetical (A-Z)"}), # New input
                "skip_first": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "load_cap": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            }
        }
    RETURN_TYPES = ("AUDIO", "AUDIO_LIST", "STRING")
    RETURN_NAMES = ("audio_batch", "audio_list", "filenames")
    FUNCTION = "load_batch"
    CATEGORY = "L3/AudioTools/IO"

    def load_batch(self, directory_path: str, file_pattern: str, sort_order: str, skip_first: int, load_cap: int):
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"ComfyAudio: The specified path is not a valid directory: {directory_path}")

        # Get initial list of file paths
        initial_file_paths = glob.glob(os.path.join(directory_path, file_pattern))

        # Filter out paths that might have disappeared between glob and sorting
        accessible_file_paths = [p for p in initial_file_paths if os.path.exists(p)]
        if len(accessible_file_paths) != len(initial_file_paths):
            print(f"ComfyAudio Warning: {len(initial_file_paths) - len(accessible_file_paths)} file(s) disappeared or became inaccessible before sorting.")

        # Apply sorting based on sort_order
        if not accessible_file_paths:
            file_paths = []
        elif sort_order == "Alphabetical (A-Z)":
            file_paths = sorted(accessible_file_paths)
        elif sort_order == "Alphabetical (Z-A)":
            file_paths = sorted(accessible_file_paths, reverse=True)
        elif sort_order == "Date/Time (Latest First)":
            try:
                # Sort by modification time, newest first
                file_paths = sorted(accessible_file_paths, key=os.path.getmtime, reverse=True)
            except Exception as e:
                print(f"ComfyAudio Warning: Error during date sorting ('Latest First'): {e}. Falling back to Alphabetical (A-Z).")
                file_paths = sorted(accessible_file_paths) # Fallback to alphabetical
        elif sort_order == "Date/Time (Oldest First)":
            try:
                # Sort by modification time, oldest first
                file_paths = sorted(accessible_file_paths, key=os.path.getmtime)
            except Exception as e:
                print(f"ComfyAudio Warning: Error during date sorting ('Oldest First'): {e}. Falling back to Alphabetical (A-Z).")
                file_paths = sorted(accessible_file_paths) # Fallback to alphabetical
        else:
            # Default or unrecognized, fall back to alphabetical A-Z
            print(f"ComfyAudio: Unrecognized sort_order '{sort_order}', defaulting to Alphabetical (A-Z).")
            file_paths = sorted(accessible_file_paths)
        
        # Logging information
        print(f"ComfyAudio: Found {len(initial_file_paths)} files matching pattern. {len(accessible_file_paths)} were accessible.")
        if accessible_file_paths:
             print(f"ComfyAudio: Sorted by '{sort_order}'. Total after sorting (before skip/cap): {len(file_paths)}. First few: {[os.path.basename(p) for p in file_paths[:5]] if file_paths else 'None'}")

        if skip_first > 0:
            if skip_first >= len(file_paths):
                return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100}, [], [],)
            file_paths = file_paths[skip_first:]
            print(f"ComfyAudio: Skipped first {skip_first} files, {len(file_paths)} remaining.")

        if load_cap != -1 and load_cap > 0:
            file_paths = file_paths[:load_cap]
            print(f"ComfyAudio: Capped loading to {len(file_paths)} files.")
        
        if not file_paths:
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100}, [], [],)

        audio_list_of_dicts = []
        waveforms_list_for_batch = []
        successful_filenames = []
        target_sr = None

        for file_path in file_paths:
            try:
                waveform, sr = torchaudio.load(file_path)
                
                if target_sr is None: target_sr = sr
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != target_sr: waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
                if waveform.dtype != torch.float32 and torch.max(torch.abs(waveform)) > 1.0:
                    waveform = waveform.to(torch.float32) / torch.iinfo(torch.int16).max
                
                unpadded_audio_dict = {"waveform": waveform.unsqueeze(0), "sample_rate": target_sr}
                audio_list_of_dicts.append(unpadded_audio_dict)
                waveforms_list_for_batch.append(waveform)
                successful_filenames.append(os.path.basename(file_path))

            except Exception as e:
                print(f"ComfyAudio Warning: Skipping file '{file_path}' due to an error: {e}")
        
        if not audio_list_of_dicts: # Should be caught by 'if not file_paths' earlier, but as a safeguard
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100}, [], [],)
            
        max_len = max(w.shape[1] for w in waveforms_list_for_batch)
        padded_waveforms = [torch.nn.functional.pad(w, (0, max_len - w.shape[1])) for w in waveforms_list_for_batch]
        batch_tensor = torch.stack(padded_waveforms)
        batch_dict = {"waveform": batch_tensor, "sample_rate": target_sr}
        
        print(f"ComfyAudio: Batch created with shape {batch_tensor.shape}. List created with {len(audio_list_of_dicts)} items.")
        print(f"ComfyAudio: Outputting {len(successful_filenames)} filenames.")

        return (batch_dict, audio_list_of_dicts, successful_filenames)
    
class GetAudioFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "get_audio"
    CATEGORY = "L3/AudioTools/IO"

    def get_audio(self, audio_list: list, index: int):
        if not audio_list:
            raise ValueError("ComfyAudio: The provided audio_list is empty.")
        
        actual_index = index % len(audio_list)
        
        return (audio_list[actual_index],)