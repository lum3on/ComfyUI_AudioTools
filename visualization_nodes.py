import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class DisplayWaveform:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "width": ("INT", {"default": 512, "min": 128, "max": 4096}), "height": ("INT", {"default": 256, "min": 128, "max": 4096}), "line_color": ("STRING", {"default": "#1f77b4"}), "bg_color": ("STRING", {"default": "#FFFFFF"}), "show_axis": ("BOOLEAN", {"default": True}),}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_waveform_image"
    CATEGORY = "L3/AudioTools/Visualization"

    def generate_waveform_image(self, audio, width, height, line_color, bg_color, show_axis):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        image_list = []
        
        # --- BATCH PROCESSING ---
        for w in w_batch:
            waveform_np = w.cpu().numpy()
            if waveform_np.ndim > 1 and waveform_np.shape[0] > 1: waveform_np = np.mean(waveform_np, axis=0)
            waveform_np = waveform_np.flatten()
            
            time_axis = np.linspace(0, len(waveform_np) / sample_rate, num=len(waveform_np))
            
            dpi = 100
            fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            ax.plot(time_axis, waveform_np, color=line_color, linewidth=1)
            ax.set_ylim([-1.05, 1.05])
            ax.set_xlim([0, len(waveform_np) / sample_rate])
            
            axis_color = 'black' if bg_color.upper() in ['#FFFFFF', 'WHITE'] else 'white'
            if show_axis:
                ax.spines['bottom'].set_color(axis_color)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_color(axis_color)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='x', colors=axis_color)
                ax.tick_params(axis='y', colors=axis_color)
                ax.set_xlabel("Time (s)", color=axis_color)
                ax.set_ylabel("Amplitude", color=axis_color)
            else:
                ax.axis('off')
                
            fig.tight_layout(pad=0)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            plt.close(fig) # IMPORTANT: Close the figure to free memory
            
            image_list.append(image_tensor)
        
        if not image_list:
            # Return an empty tensor if no images were generated
            return (torch.zeros((0, height, width, 3)),)
            
        # Stack the list of image tensors into a single batch tensor
        return (torch.cat(image_list, dim=0),)


class CompareWaveforms:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio_a": ("AUDIO",), "audio_b": ("AUDIO",), "width": ("INT", {"default": 512, "min": 128, "max": 4096}), "height": ("INT", {"default": 256, "min": 128, "max": 4096}), "color_a": ("STRING", {"default": "#1f77b4"}), "color_b": ("STRING", {"default": "#ff7f0e"}), "bg_color": ("STRING", {"default": "#FFFFFF"}), "show_axis": ("BOOLEAN", {"default": True}),}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_comparison_image"
    CATEGORY = "L3/AudioTools/Visualization"
    
    def _prep_waveform(self, w, sr):
        waveform_np = w.cpu().numpy()
        if waveform_np.ndim > 1 and waveform_np.shape[0] > 1: waveform_np = np.mean(waveform_np, axis=0)
        return waveform_np.flatten()
        
    def generate_comparison_image(self, audio_a, audio_b, width, height, color_a, color_b, bg_color, show_axis):
        w_batch_a, sr_a = audio_a["waveform"], audio_a["sample_rate"]
        w_batch_b, sr_b = audio_b["waveform"], audio_b["sample_rate"]
        image_list = []

        batch_size = min(len(w_batch_a), len(w_batch_b))
        if len(w_batch_a) != len(w_batch_b):
            print(f"ComfyAudio Warning: CompareWaveforms received batches of different sizes ({len(w_batch_a)} vs {len(w_batch_b)}). Comparing first {batch_size} items.")

        for i in range(batch_size):
            waveform_a_np = self._prep_waveform(w_batch_a[i], sr_a)
            waveform_b_np = self._prep_waveform(w_batch_b[i], sr_b)
            
            max_len = max(len(waveform_a_np), len(waveform_b_np))
            max_sr = max(sr_a, sr_b)
            duration = max_len / max_sr
            time_axis_a = np.linspace(0, len(waveform_a_np) / sr_a, num=len(waveform_a_np))
            time_axis_b = np.linspace(0, len(waveform_b_np) / sr_b, num=len(waveform_b_np))
            
            dpi = 100
            fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
            
            # --- THE FIX: Corrected typo from bg_.color to bg_color ---
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            ax.plot(time_axis_a, waveform_a_np, color=color_a, linewidth=1, label="Audio A", alpha=0.7)
            ax.plot(time_axis_b, waveform_b_np, color=color_b, linewidth=1, label="Audio B", alpha=0.7)
            ax.set_ylim([-1.05, 1.05])
            ax.set_xlim([0, duration])
            
            axis_color = 'black' if bg_color.upper() in ['#FFFFFF', 'WHITE'] else 'white'
            if show_axis:
                ax.spines['bottom'].set_color(axis_color)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_color(axis_color)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='x', colors=axis_color)
                ax.tick_params(axis='y', colors=axis_color)
                ax.set_xlabel("Time (s)", color=axis_color)
                ax.set_ylabel("Amplitude", color=axis_color)
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
            
            image_list.append(image_tensor)

        if not image_list:
            return (torch.zeros((0, height, width, 3)),)
            
        return (torch.cat(image_list, dim=0),)
    

class ShowAudioInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",),}}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "show_info"
    OUTPUT_NODE = True
    CATEGORY = "L3/AudioTools/Visualization"

    def show_info(self, audio: dict):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None:
            text = "Invalid Audio Input: Not a valid ComfyAudio signal."
        else:
            # Display info about the first item in the batch for simplicity
            batch_size, channels, num_samples = waveform.shape
            duration_seconds = num_samples / sample_rate
            bit_depth = 16 
            bitrate_kbps = (sample_rate * bit_depth * channels) / 1000
            text = (
                f"--- Batch Info ---\n"
                f"Batch Size: {batch_size}\n"
                f"Sample Rate: {sample_rate} Hz\n"
                f"\n--- Info for Item 0 ---\n"
                f"Duration: {duration_seconds:.2f} seconds\n"
                f"Channels: {channels}\n"
                f"Samples: {num_samples}\n"
                f"Min Value: {waveform[0].min():.4f}\n"
                f"Max Value: {waveform[0].max():.4f}\n"
                f"Data Type: {waveform.dtype}"
            )
        
        print("\n--- Audio Info ---\n" + text)
        return {"ui": {"text": [text]}, "result": (text,)}