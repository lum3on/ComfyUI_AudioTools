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
                "skip_first": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "load_cap": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            }
        }
    # --- UPDATED: ADD THE NEW AUDIO_LIST OUTPUT ---
    RETURN_TYPES = ("AUDIO", "AUDIO_LIST")
    RETURN_NAMES = ("audio_batch", "audio_list")
    FUNCTION = "load_batch"
    CATEGORY = "L3/AudioTools/IO"

    def load_batch(self, directory_path: str, file_pattern: str, skip_first: int, load_cap: int):
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"ComfyAudio: The specified path is not a valid directory: {directory_path}")

        file_paths = sorted(glob.glob(os.path.join(directory_path, file_pattern)))
        print(f"ComfyAudio: Found {len(file_paths)} total files.")
        
        if skip_first > 0:
            if skip_first >= len(file_paths):
                return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100}, [],) # Return empty for both
            file_paths = file_paths[skip_first:]
            print(f"ComfyAudio: Skipped first {skip_first} files, {len(file_paths)} remaining.")

        if load_cap != -1 and load_cap > 0:
            file_paths = file_paths[:load_cap]
            print(f"ComfyAudio: Capped loading to {len(file_paths)} files.")
        
        if not file_paths:
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100}, [],)

        # This will hold the unpadded audio dicts for the list output
        audio_list_of_dicts = []
        waveforms_list_for_batch = []
        target_sr = None

        for file_path in file_paths:
            try:
                waveform, sr = torchaudio.load(file_path)
                
                if target_sr is None: target_sr = sr
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != target_sr: waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
                if waveform.dtype != torch.float32 and torch.max(torch.abs(waveform)) > 1.0:
                    waveform = waveform.to(torch.float32) / torch.iinfo(torch.int16).max
                
                # --- NEW: Create the individual, unpadded audio object for the list ---
                # The waveform must have a batch dimension of 1 to be a valid AUDIO type
                unpadded_audio_dict = {"waveform": waveform.unsqueeze(0), "sample_rate": target_sr}
                audio_list_of_dicts.append(unpadded_audio_dict)
                
                # Add the raw waveform (without extra batch dim) to the batch list
                waveforms_list_for_batch.append(waveform)
            except Exception as e:
                print(f"ComfyAudio Warning: Skipping file '{file_path}' due to an error: {e}")
        
        if not audio_list_of_dicts:
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": 44100}, [],)
            
        # Create the padded batch tensor (existing logic)
        max_len = max(w.shape[1] for w in waveforms_list_for_batch)
        padded_waveforms = [torch.nn.functional.pad(w, (0, max_len - w.shape[1])) for w in waveforms_list_for_batch]
        batch_tensor = torch.stack(padded_waveforms)
        batch_dict = {"waveform": batch_tensor, "sample_rate": target_sr}
        
        print(f"ComfyAudio: Batch created with shape {batch_tensor.shape}. List created with {len(audio_list_of_dicts)} items.")

        # --- UPDATED: Return both the batch and the list ---
        return (batch_dict, audio_list_of_dicts)
    
class GetAudioFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    # This is a custom type. ComfyUI will see it as a valid connection point.
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "get_audio"
    CATEGORY = "L3/AudioTools/IO"

    def get_audio(self, audio_list: list, index: int):
        if not audio_list:
            raise ValueError("ComfyAudio: The provided audio_list is empty.")
        
        # Use modulo to ensure the index is always valid and can be used for looping
        actual_index = index % len(audio_list)
        
        return (audio_list[actual_index],)