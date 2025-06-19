# File: ai_nodes.py (Final Version with Resampling and Task Selection)

import torch
import numpy as np
import torchaudio

# Lazy-loading heavy libraries
def import_demucs_modules():
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        return get_model, apply_model
    except ImportError as e:
        print(f"ComfyAudio: Failed to import demucs modules. Original error: {e}")
        raise ImportError(
            "Demucs or one of its dependencies is not installed correctly. "
            "Please ensure you have run the installation command for the requirements.txt file."
        )

def import_whisper():
    try:
        import whisper
        return whisper
    except ImportError:
        raise ImportError("Whisper not installed. Please run 'pip install -U openai-whisper'")

class StemSeparator:
    MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
    def __init__(self):
        self.models = {}

    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "model_name": (s.MODELS,), } }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "bass", "drums", "other")
    FUNCTION = "separate"
    CATEGORY = "L3/AudioTools/AI"

    def separate(self, audio: dict, model_name: str):
        get_model, apply_model = import_demucs_modules()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_name not in self.models:
            model = get_model(name=model_name)
            model.to(device)
            self.models[model_name] = model
        
        model = self.models[model_name]
        waveform = audio["waveform"][0]
        sample_rate = audio["sample_rate"]
        
        if sample_rate != model.samplerate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.samplerate)
            waveform = resampler(waveform)

        sources = apply_model(model, waveform.unsqueeze(0), device=device, progress=True)[0]
        stem_map = {'drums': 0, 'bass': 1, 'other': 2, 'vocals': 3}
        def create_output(stem_tensor):
            return {"waveform": stem_tensor.unsqueeze(0), "sample_rate": model.samplerate}
        return (create_output(sources[stem_map['vocals']]), create_output(sources[stem_map['bass']]), create_output(sources[stem_map['drums']]), create_output(sources[stem_map['other']]))

class SpeechDenoise:
    MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
    def __init__(self):
        self.models = {}

    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "model_name": (s.MODELS, {"default": "htdemucs"}), } }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "denoise"
    CATEGORY = "L3/AudioTools/AI"

    def denoise(self, audio: dict, model_name: str):
        get_model, apply_model = import_demucs_modules()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_name not in self.models:
            model = get_model(name=model_name)
            model.to(device)
            self.models[model_name] = model

        model = self.models[model_name]
        processed_list = []
        for w in audio["waveform"]:
            if audio["sample_rate"] != model.samplerate:
                resampler = torchaudio.transforms.Resample(orig_freq=audio["sample_rate"], new_freq=model.samplerate)
                w = resampler(w)
            sources = apply_model(model, w.unsqueeze(0), device=device, progress=True)[0]
            vocals_stem = sources[3]
            processed_list.append(vocals_stem)
        processed_batch = torch.stack(processed_list)
        return ({"waveform": processed_batch, "sample_rate": model.samplerate},)

class SpeechToTextWhisper:
    LANGUAGES = ["Auto-Detect", "English", "German", "Spanish", "French", "Italian", "Japanese", "Chinese", "Russian", "Korean", "Portuguese"]
    LANGUAGE_CODES = { "Auto-Detect": None, "English": "en", "German": "de", "Spanish": "es", "French": "fr", "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Russian": "ru", "Korean": "ko", "Portuguese": "pt" }
    MODELS = ["tiny", "base", "small", "medium", "large-v2"]
    _models = {}

    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "audio": ("AUDIO",), 
                "model_size": (s.MODELS, {"default": "base"}),
                "language": (s.LANGUAGES, {"default": "Auto-Detect"}),
                # --- FIX 2: Add a dropdown to explicitly set the task ---
                "task": (["Transcribe", "Translate to English"], {"default": "Transcribe"}),
            } 
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe"
    CATEGORY = "L3/AudioTools/AI"

    def transcribe(self, audio: dict, model_size: str, language: str, task: str):
        whisper = import_whisper()
        if model_size not in self._models:
            self._models[model_size] = whisper.load_model(model_size)
        model = self._models[model_size]

        lang_code = self.LANGUAGE_CODES.get(language, None)
        task_str = task.lower().split(' ')[0] # Converts "Translate to English" to "translate"

        print(f"ComfyAudio: Transcribing with language='{language}' (code: {lang_code}) and task='{task_str}'")

        waveform = audio["waveform"][0]
        sample_rate = audio["sample_rate"]
        
        # --- FIX 1: Automatically resample the audio to 16kHz, which Whisper requires ---
        WHISPER_SR = 16000
        if sample_rate != WHISPER_SR:
            print(f"ComfyAudio: Resampling for Whisper from {sample_rate}Hz to {WHISPER_SR}Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=WHISPER_SR)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        audio_np = waveform.cpu().numpy().astype(np.float32)
        
        # --- UPDATED: Pass both the language code and the explicit task to the model ---
        result = model.transcribe(audio_np, language=lang_code, task=task_str, fp16=torch.cuda.is_available())
        
        return (result["text"],)