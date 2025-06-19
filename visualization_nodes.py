import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import folder_paths

class DisplayWaveform:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "width": ("INT", {"default": 1024, "min": 128, "max": 4096}), "height": ("INT", {"default": 512, "min": 128, "max": 4096}), "line_color": ("STRING", {"default": "#1f77b4"}), "bg_color": ("STRING", {"default": "#FFFFFF"}), "show_axis": ("BOOLEAN", {"default": True}),}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_waveform_image"
    CATEGORY = "L3/AudioTools/Analysis"

    def generate_waveform_image(self, audio, width, height, line_color, bg_color, show_axis):
        waveform, sample_rate = None, None
        if isinstance(audio, dict) and "waveform" in audio:
            waveform, sample_rate = audio["waveform"][0], audio["sample_rate"]
        elif isinstance(audio, str):
            try:
                audio_path = folder_paths.get_annotated_filepath(audio)
                waveform, sample_rate = torchaudio.load(audio_path)
                if torch.max(torch.abs(waveform)) > 1.0: waveform = waveform / torch.iinfo(torch.int16).max
            except Exception as e: raise e
        if waveform is None: raise TypeError(f"Unsupported input type: {type(audio)}")
        
        waveform_np = waveform.cpu().numpy()
        if waveform_np.ndim > 1 and waveform_np.shape[0] > 1: waveform_np = np.mean(waveform_np, axis=0)
        waveform_np = waveform_np.flatten()
        
        time_axis = np.linspace(0, len(waveform_np) / sample_rate, num=len(waveform_np))
        
        # --- THE FIX: Assign dpi on its own line ---
        dpi = 100
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.plot(time_axis, waveform_np, color=line_color, linewidth=1)
        ax.set_ylim([-1.05, 1.05])
        ax.set_xlim([0, len(waveform_np) / sample_rate])
        
        if show_axis:
            ax.spines['bottom'].set_color('black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color('black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', colors='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.tick_params(axis='y', colors='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.set_xlabel("Time (s)", color='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.set_ylabel("Amplitude", color='black' if bg_color.upper() == '#FFFFFF' else 'white')
        else:
            ax.axis('off')
            
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        plt.close(fig)
        
        return (image_tensor,)

class CompareWaveforms:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio_a": ("AUDIO",), "audio_b": ("AUDIO",), "width": ("INT", {"default": 1024, "min": 128, "max": 4096}), "height": ("INT", {"default": 512, "min": 128, "max": 4096}), "color_a": ("STRING", {"default": "#1f77b4"}), "color_b": ("STRING", {"default": "#ff7f0e"}), "bg_color": ("STRING", {"default": "#FFFFFF"}), "show_axis": ("BOOLEAN", {"default": True}),}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_comparison_image"
    CATEGORY = "L3/AudioTools/Analysis"
    
    def _load_and_prep_audio(self, audio_input):
        waveform, sample_rate = None, None
        if isinstance(audio_input, dict) and "waveform" in audio_input:
            waveform, sample_rate = audio_input["waveform"][0], audio_input["sample_rate"]
        elif isinstance(audio_input, str):
            try:
                audio_path = folder_paths.get_annotated_filepath(audio_input)
                waveform, sample_rate = torchaudio.load(audio_path)
                if torch.max(torch.abs(waveform)) > 1.0: waveform = waveform / torch.iinfo(torch.int16).max
            except Exception as e: raise e
        if waveform is None: raise TypeError(f"Unsupported input type: {type(audio_input)}")
        
        waveform_np = waveform.cpu().numpy()
        if waveform_np.ndim > 1 and waveform_np.shape[0] > 1: waveform_np = np.mean(waveform_np, axis=0)
        return waveform_np.flatten(), sample_rate
        
    def generate_comparison_image(self, audio_a, audio_b, width, height, color_a, color_b, bg_color, show_axis):
        waveform_a_np, sr_a = self._load_and_prep_audio(audio_a)
        waveform_b_np, sr_b = self._load_and_prep_audio(audio_b)
        
        max_len, max_sr = max(len(waveform_a_np), len(waveform_b_np)), max(sr_a, sr_b)
        duration = max_len / max_sr
        time_axis_a = np.linspace(0, len(waveform_a_np) / sr_a, num=len(waveform_a_np))
        time_axis_b = np.linspace(0, len(waveform_b_np) / sr_b, num=len(waveform_b_np))
        
        # --- THE FIX: Assign dpi on its own line ---
        dpi = 100
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.plot(time_axis_a, waveform_a_np, color=color_a, linewidth=1, label="Audio A", alpha=0.7)
        ax.plot(time_axis_b, waveform_b_np, color=color_b, linewidth=1, label="Audio B", alpha=0.7)
        ax.set_ylim([-1.05, 1.05])
        ax.set_xlim([0, duration])
        
        if show_axis:
            ax.spines['bottom'].set_color('black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color('black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', colors='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.tick_params(axis='y', colors='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.set_xlabel("Time (s)", color='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.set_ylabel("Amplitude", color='black' if bg_color.upper() == '#FFFFFF' else 'white')
            ax.legend()
        else:
            ax.axis('off')
            
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        plt.close(fig)
        
        return (image_tensor,)

class ShowAudioInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",),}}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "show_info"
    OUTPUT_NODE = True
    CATEGORY = "L3/AudioTools/Analysis"

    def show_info(self, audio: dict):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None:
            text = "Invalid Audio Input: Not a valid ComfyAudio signal."
        else:
            batch_size, channels, num_samples = waveform.shape
            duration_seconds = num_samples / sample_rate
            bit_depth = 16 
            bitrate_kbps = (sample_rate * bit_depth * channels) / 1000
            text = (
                f"Sample Rate: {sample_rate} Hz\n"
                f"Duration: {duration_seconds:.2f} seconds\n"
                f"Batch Size: {batch_size}\n"
                f"Channels: {channels}\n"
                f"Samples: {num_samples}\n"
                f"Min Value: {waveform.min():.4f}\n"
                f"Max Value: {waveform.max():.4f}\n"
                f"Data Type: {waveform.dtype}\n"
                f"Est. Bitrate: {bitrate_kbps:.0f} kbps (at 16-bit)"
            )
        
        print("\n--- Audio Info ---\n" + text)
        return {"ui": {"text": [text]}, "result": (text,)}