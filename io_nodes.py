# File: io_nodes.py (Corrected and Enhanced)

import torch
import torchaudio
import folder_paths
import os
import time
import hashlib

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        # The 'audio' type is a custom widget provided by ComfyUI that gives us the file path.
        # It's not a real file upload, but a file picker from the input directory.
        # To allow true uploads, you'd need a different setup or a pre-processing step.
        # For now, we'll assume files are in the ComfyUI 'input' folder.
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "audio_file": (sorted(files), {"audio_upload": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"
    CATEGORY = "L3/AudioTools/IO"

    def load_audio(self, audio_file):
        # The widget now handles the upload and provides a path relative to the input dir
        file_path = folder_paths.get_annotated_filepath(audio_file)
        print(f"ComfyAudio: Loading audio from {file_path}")
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono for wider compatibility with other nodes
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return ((waveform, sample_rate),)

class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "ComfyAudio/output"}),
                "format": (["wav", "mp3", "flac"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "L3/AudioTools/IO"

    def save_audio(self, audio, filename_prefix, format):
        waveform, sample_rate = audio
        
        # Create the full output path
        full_output_folder, filename, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
        # Append the correct extension
        file = f"{filename}.{format}"
        file_path = os.path.join(full_output_folder, file)

        print(f"ComfyAudio: Saving audio to {file_path}")

        # *** FIX: Create the directory if it doesn't exist ***
        os.makedirs(full_output_folder, exist_ok=True)

        # Ensure waveform is on CPU and in a suitable format for saving
        waveform_cpu = waveform.cpu()
        
        torchaudio.save(file_path, waveform_cpu, sample_rate)
        
        # Provide the result for the UI
        return {"ui": {"audio": [{"filename": file, "subfolder": os.path.relpath(full_output_folder, self.output_dir), "type": "output"}]}}


class PreviewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "audio": ("AUDIO",) }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_audio"
    OUTPUT_NODE = True
    CATEGORY = "L3/AudioTools/IO"

    def preview_audio(self, audio):
        waveform, sample_rate = audio
        
        # Save audio to a temporary file
        temp_dir = folder_paths.get_temp_directory()
        
        # Use a hash of the audio data to name the file, preventing duplicates
        waveform_hash = hashlib.sha256(waveform.cpu().numpy().tobytes()).hexdigest()
        filename = f"preview_{waveform_hash[:10]}.wav"
        file_path = os.path.join(temp_dir, filename)

        # Only save if it doesn't already exist
        if not os.path.exists(file_path):
            print(f"ComfyAudio: Saving temporary preview to {file_path}")
            torchaudio.save(file_path, waveform.cpu(), sample_rate)

        # *** FIX: Return an HTML audio player for the UI ***
        # The URL is relative to the ComfyUI root. /temp serves files from the temp directory.
        return {"ui": {"html": [f'<audio src="/temp/{filename}" controls loop></audio>']}}