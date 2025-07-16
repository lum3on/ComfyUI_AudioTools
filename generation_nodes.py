import torch
import torchaudio
import pyttsx3
import os
import folder_paths
import hashlib

class TextToSpeechNode:
    _engine = None
    _voices = None

    @classmethod
    def INPUT_TYPES(s):
        # Lazily initialize the engine to get voice info
        if s._engine is None:
            s._engine = pyttsx3.init()
            s._voices = s._engine.getProperty('voices')
            print("--- ComfyAudio TTS Voices ---")
            for i, v in enumerate(s._voices):
                print(f"  Index {i}: {v.name} ({v.id})")
            print("-----------------------------")

        voice_names = [f"Voice {i}: {v.name}" for i, v in enumerate(s._voices)]
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of the text to speech node.", "tooltip": "The text to be converted into speech."}),
                "voice_index": (voice_names, {"tooltip": "The system voice to use for synthesis. Voice options depend on your operating system."}),
                "rate": ("INT", {"default": 175, "min": 50, "max": 350, "step": 5, "tooltip": "The speaking rate in words per minute."}),
                "volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "The volume of the generated audio (0.0 to 1.0)."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "synthesize"
    CATEGORY = "⚪Lum3on/AudioTools/Generation"

    def synthesize(self, text, voice_index, rate, volume):
        # The engine is already initialized by INPUT_TYPES
        engine = self._engine
        
        # The voice_index is a string like "Voice 0: Microsoft David...", we need to parse the index
        selected_index = int(voice_index.split(':')[0].replace('Voice ', ''))

        if 0 <= selected_index < len(self._voices):
            engine.setProperty('voice', self._voices[selected_index].id)
        
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)

        # Use a hash of the input to create a unique temporary filename
        input_hash = hashlib.sha256((text + voice_index + str(rate) + str(volume)).encode()).hexdigest()
        temp_dir = folder_paths.get_temp_directory()
        temp_path = os.path.join(temp_dir, f"tts_{input_hash[:10]}.wav")

        print(f"ComfyAudio TTS: Synthesizing to temporary file: {temp_path}")
        
        # Synthesize and save the audio to the temporary file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        # Load the synthesized audio back into a tensor
        try:
            waveform, sample_rate = torchaudio.load(temp_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load synthesized audio file. Error: {e}")

        # Ensure audio is in float format and normalized
        if torch.max(torch.abs(waveform)) > 1.0:
            waveform = waveform / torch.iinfo(torch.int16).max
            
        # The native ComfyUI format is a dictionary
        # The pyttsx3 library often produces mono, but we prepare for stereo just in case
        output_audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        
        return (output_audio,)