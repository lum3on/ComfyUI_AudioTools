# File: ai_nodes.py (With SRT Generation)

import torch
import numpy as np
import torchaudio
import math

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

# --- SRT Helper Functions ---
def format_time_srt(seconds: float) -> str:
    """Converts seconds to HH:MM:SS,mmm format for SRT."""
    if seconds < 0: seconds = 0.0
    millis = round((seconds - math.floor(seconds)) * 1000)
    
    if millis >= 1000:
        seconds += 1
        millis = 0
        
    total_seconds = int(math.floor(seconds))
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def _split_text_into_lines(text_segment: str, max_char_len: int, max_lines_count: int) -> list[str]:
    """
    Splits a given text segment into lines, respecting max_char_len per line
    and max_lines_count for the whole block.
    """
    words = text_segment.strip().split()
    lines_output = []
    current_line_text = ""

    for word in words:
        if not current_line_text:
            current_line_text = word
        elif len(current_line_text) + 1 + len(word) <= max_char_len:
            current_line_text += " " + word
        else:
            if len(lines_output) < max_lines_count:
                lines_output.append(current_line_text)
                current_line_text = word
            else:
                current_line_text = ""
                break 
    
    if current_line_text and len(lines_output) < max_lines_count:
        lines_output.append(current_line_text)
    
    return lines_output


def _generate_srt_from_whisper_result(
    whisper_result: dict, 
    max_line_len: int, 
    max_lines_per_block: int, 
    max_block_duration_sec: float 
) -> str:
    """
    Generates an SRT formatted string from Whisper transcription result,
    focusing on robustly handling word-level timestamps.
    """
    srt_blocks_list = []
    srt_block_idx = 1

    if "segments" not in whisper_result or not whisper_result["segments"]:
        print("ComfyAudio SRT: No segments found in Whisper result.")
        return "" 

    for segment in whisper_result["segments"]:
        segment_words_list = segment.get("words")
        
        if not segment_words_list:
            seg_text = segment.get("text", "").strip()
            if not seg_text: 
                continue

            start_t = segment.get("start", 0.0)
            end_t = segment.get("end", start_t + 0.5)
            if end_t <= start_t: end_t = start_t + 0.5 

            lines = _split_text_into_lines(seg_text, max_line_len, max_lines_per_block)
            if lines:
                block_text = "\n".join(lines)
                block = (
                    f"{srt_block_idx}\n"
                    f"{format_time_srt(start_t)} --> {format_time_srt(end_t)}\n"
                    f"{block_text}"
                )
                srt_blocks_list.append(block)
                srt_block_idx += 1
            continue


        current_block_word_objects = []

        for i, word_info in enumerate(segment_words_list):
            if not isinstance(word_info, dict) or \
               "word" not in word_info or \
               "start" not in word_info or \
               "end" not in word_info:
                print(f"ComfyAudio SRT: Skipping malformed word_info: {word_info}")
                continue

            word_text_from_whisper = word_info["word"].strip() 
            if not word_text_from_whisper:
                continue

            current_word_obj = {
                "text": word_text_from_whisper,
                "start": word_info["start"],
                "end": word_info["end"]
            }

            if not current_block_word_objects:
                current_block_word_objects.append(current_word_obj)
            else:
                block_start_time = current_block_word_objects[0]["start"]
                potential_block_end_time = current_word_obj["end"] 
                
                if (potential_block_end_time - block_start_time) > max_block_duration_sec:
                    
                    text_for_srt_block = " ".join(w["text"] for w in current_block_word_objects)
                    start_time_for_srt_block = current_block_word_objects[0]["start"]
                    end_time_for_srt_block = current_block_word_objects[-1]["end"]

                    lines = _split_text_into_lines(text_for_srt_block, max_line_len, max_lines_per_block)
                    if lines:
                        block_text = "\n".join(lines)
                        block = (
                            f"{srt_block_idx}\n"
                            f"{format_time_srt(start_time_for_srt_block)} --> {format_time_srt(end_time_for_srt_block)}\n"
                            f"{block_text}"
                        )
                        srt_blocks_list.append(block)
                        srt_block_idx += 1
                    
                    current_block_word_objects = [current_word_obj]
                else:
                    current_block_word_objects.append(current_word_obj)

            if i == len(segment_words_list) - 1:
                if current_block_word_objects: 
                    text_for_srt_block = " ".join(w["text"] for w in current_block_word_objects)
                    start_time_for_srt_block = current_block_word_objects[0]["start"]
                    end_time_for_srt_block = current_block_word_objects[-1]["end"] 

                    lines = _split_text_into_lines(text_for_srt_block, max_line_len, max_lines_per_block)
                    if lines:
                        block_text = "\n".join(lines)
                        block = (
                            f"{srt_block_idx}\n"
                            f"{format_time_srt(start_time_for_srt_block)} --> {format_time_srt(end_time_for_srt_block)}\n"
                            f"{block_text}"
                        )
                        srt_blocks_list.append(block)
                        srt_block_idx += 1
                    current_block_word_objects = [] 
                    
    if not srt_blocks_list:
        print("ComfyAudio SRT: No SRT blocks were generated.")
        
    return "\n\n".join(srt_blocks_list)

class StemSeparator:
    MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
    def __init__(self):
        self.models = {}

    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
                    "audio": ("AUDIO", {"tooltip": "The audio clip to be separated into stems."}), 
                    "model_name": (s.MODELS, {"tooltip": "The Demucs model to use for separation. 'htdemucs_ft' is a good general-purpose choice."}), } }

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
        return { "required": { 
                        "audio": ("AUDIO", {"tooltip": "The audio clip containing speech to be denoised."}), 
                        "model_name": (s.MODELS, {"default": "htdemucs", "tooltip": "The Demucs model to use for isolating vocals. It will remove non-vocal sounds."}), } }

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
            vocals_stem = sources[3] # Vocals stem
            processed_list.append(vocals_stem)
        processed_batch = torch.stack(processed_list)
        return ({"waveform": processed_batch, "sample_rate": model.samplerate},)


class SpeechToTextWhisper:
    LANGUAGES = ["Auto-Detect", "English", "German", "Spanish", "French", "Italian", "Japanese", "Chinese", "Russian", "Korean", "Portuguese"]
    LANGUAGE_CODES = { "Auto-Detect": None, "English": "en", "German": "de", "Spanish": "es", "French": "fr", "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Russian": "ru", "Korean": "ko", "Portuguese": "pt" }
    MODELS = ["tiny", "base", "small", "medium", "large-v2"]
    _models = {}

    @classmethod
    def INPUT_TYPES(cls): # Changed 's' to 'cls' for @classmethod
        return { 
            "required": { 
                "audio": ("AUDIO", {"tooltip": "The audio clip to be transcribed."}), 
                "model_size": (cls.MODELS, {"default": "base", "tooltip": "Whisper model size."}),
                "language": (cls.LANGUAGES, {"default": "Auto-Detect", "tooltip": "Language of the speech."}),
                "task": (["Transcribe", "Translate to English"], {"default": "Transcribe", "tooltip": "Transcription or translation task."}),
            },
            "optional": { # New SRT options
                "generate_srt": ("BOOLEAN", {"default": False, "label_on": "ENABLED", "label_off": "DISABLED", "tooltip": "Generate SRT subtitle output."}),
                "srt_max_line_len": ("INT", {"default": 200, "min": 10, "max": 200, "step": 1, "tooltip": "Max characters per SRT line."}),
                "srt_max_lines": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1, "tooltip": "Max lines per SRT block."}),
                "srt_max_duration_sec": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5, "tooltip": "Max duration (seconds) of an SRT block."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING") # Plain text, SRT text
    RETURN_NAMES = ("text", "srt_text")
    FUNCTION = "transcribe"
    CATEGORY = "L3/AudioTools/AI"

    def transcribe(self, audio: dict, model_size: str, language: str, task: str, **kwargs): # Added **kwargs
        whisper_lib = import_whisper()
        if model_size not in self._models:
            self._models[model_size] = whisper_lib.load_model(model_size)
        model = self._models[model_size]

        lang_code = self.LANGUAGE_CODES.get(language, None)
        task_str = task.lower().split(' ')[0] 

        print(f"ComfyAudio: Transcribing with model='{model_size}', language='{language}' (code: {lang_code}), task='{task_str}'")

        waveform = audio["waveform"][0] 
        sample_rate = audio["sample_rate"]
        
        WHISPER_SR = 16000
        if sample_rate != WHISPER_SR:
            print(f"ComfyAudio: Resampling for Whisper from {sample_rate}Hz to {WHISPER_SR}Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=WHISPER_SR)
            waveform = resampler(waveform)

        if waveform.nelement() == 0:
            print("ComfyAudio: SpeechToTextWhisper received empty audio waveform after resampling. Returning empty strings.")
            return ("", "",)
        
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0) 
        elif waveform.ndim > 1 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        audio_np = waveform.cpu().numpy().astype(np.float32)
        if audio_np.ndim > 1: audio_np = audio_np.flatten()
        audio_np = np.ascontiguousarray(audio_np)

        if audio_np.size == 0:
            print("ComfyAudio: SpeechToTextWhisper audio_np is empty after processing. Returning empty strings.")
            return ("", "",)

        # Get SRT generation options from kwargs
        generate_srt_flag = kwargs.get("generate_srt", False)
        srt_max_line_len = kwargs.get("srt_max_line_len", 42)
        srt_max_lines = kwargs.get("srt_max_lines", 2)
        srt_max_duration_sec = kwargs.get("srt_max_duration_sec", 7.0)
        
        transcribe_args = {
            "task": task_str,
            "fp16": False, 
            "beam_size": 1, 
            "best_of": 1,
            "word_timestamps": True if generate_srt_flag else False # Crucial for SRT
        }
        if lang_code is not None:
            transcribe_args["language"] = lang_code
        
        print(f"ComfyAudio: Whisper `model.transcribe` arguments: {transcribe_args}")
        
        try:
            result = model.transcribe(audio_np, **transcribe_args)
        except Exception as e:
            print(f"ComfyAudio: Error during Whisper transcription: {e}")
            print(f"ComfyAudio: audio_np shape: {audio_np.shape}, dtype: {audio_np.dtype}, min: {audio_np.min() if audio_np.size > 0 else 'N/A'}, max: {audio_np.max() if audio_np.size > 0 else 'N/A'}")
            raise e
            
        plain_text_output = result.get("text", "").strip()
        srt_output_str = ""

        if generate_srt_flag:
            print(f"ComfyAudio: Generating SRT with max_line_len={srt_max_line_len}, max_lines={srt_max_lines}, max_duration={srt_max_duration_sec}s.")
            if "segments" not in result or not result["segments"]:
                print("ComfyAudio: SRT requested, but no segments found in Whisper result.")
            elif transcribe_args["word_timestamps"] and (not result["segments"] or "words" not in result["segments"][0]):
                 print("ComfyAudio: SRT requested with word_timestamps=True, but 'words' key not found in first segment. Cannot generate detailed SRT.")
            else:
                srt_output_str = _generate_srt_from_whisper_result(
                    result, 
                    srt_max_line_len, 
                    srt_max_lines, 
                    srt_max_duration_sec
                )
                if not srt_output_str:
                    print("ComfyAudio: SRT generation resulted in an empty string.")
        
        return (plain_text_output, srt_output_str,)