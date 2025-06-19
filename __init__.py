from .processing_nodes import (AmplifyGain, NormalizeAudio, MixAudio, RemoveSilence, 
                               DeEsser, DePlosive, ParametricEQ, VocalCompressor, 
                               Reverb, Delay, PitchTime, TrimAudio,
                               ConcatenateAudio, StereoPanner, DeHum, NoiseGate,
                               LoudnessMeter, BPMDetector, AudioReactiveParameter)
from .ai_nodes import StemSeparator, SpeechDenoise, SpeechToTextWhisper
from .visualization_nodes import DisplayWaveform, CompareWaveforms, ShowAudioInfo # New import
from .generation_nodes import TextToSpeechNode

NODE_CLASS_MAPPINGS = {
    # Generation Nodes
    "TextToSpeech": TextToSpeechNode,

    # Processing Nodes
    "AudioAmplify": AmplifyGain,
    "AudioNormalize": NormalizeAudio,
    "AudioMix": MixAudio,
    "AudioTrim": TrimAudio,
    "AudioRemoveSilence": RemoveSilence,
    "AudioDeEsser": DeEsser,
    "AudioDePlosive": DePlosive,
    "AudioParametricEQ": ParametricEQ,
    "AudioVocalCompressor": VocalCompressor,
    "AudioDeHum": DeHum,              # New
    "AudioNoiseGate": NoiseGate,        # New
    
    # Effects Nodes
    "AudioReverb": Reverb,
    "AudioDelay": Delay,
    "AudioPitchTime": PitchTime,

    # Utility Nodes (New Category)
    "AudioConcatenate": ConcatenateAudio, # New
    "AudioStereoPan": StereoPanner,     # New

    # AI Nodes
    "AudioStemSeparate": StemSeparator,
    "AudioSpeechDenoise": SpeechDenoise,
    "AudioSpeechToTextWhisper": SpeechToTextWhisper,

    # Analysis & Reactive Nodes (New Category)
    "AudioLoudnessMeter": LoudnessMeter, # New
    "AudioBPMDetector": BPMDetector,     # New
    "AudioReactiveParam": AudioReactiveParameter, # New

    # Visualization Nodes
    "AudioDisplayWaveform": DisplayWaveform,
    "AudioCompareWaveforms": CompareWaveforms,
    "AudioShowInfo": ShowAudioInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Generation Nodes
    "TextToSpeech": "Text to Speech",

    # Processing Nodes
    "AudioAmplify": "Amplify / Gain",
    "AudioNormalize": "Normalize Audio",
    "AudioMix": "Mix Audio Tracks",
    "AudioTrim": "Trim Audio",
    "AudioRemoveSilence": "Remove Silence",
    "AudioDeEsser": "De-Esser",
    "AudioDePlosive": "De-Plosive (Low Cut)",
    "AudioParametricEQ": "Parametric EQ for Voice",
    "AudioVocalCompressor": "Vocal Compressor",
    "AudioDeHum": "De-Hum (50/60Hz)", # New
    "AudioNoiseGate": "Noise Gate",     # New
    
    # Effects Nodes
    "AudioReverb": "Reverb",
    "AudioDelay": "Delay / Echo",
    "AudioPitchTime": "Pitch Shift / Time Stretch",

    # Utility Nodes (New Category)
    "AudioConcatenate": "Concatenate Audio", # New
    "AudioStereoPan": "Stereo Panner",      # New

    # AI Nodes
    "AudioStemSeparate": "Stem Separator (AI)",
    "AudioSpeechDenoise": "Speech Denoise (AI)",
    "AudioSpeechToTextWhisper": "Speech-to-Text (Whisper)",

    # Analysis & Reactive Nodes (New Category)
    "AudioLoudnessMeter": "Loudness Meter (LUFS)", # New
    "AudioBPMDetector": "BPM Detector / Reactive", # New
    "AudioReactiveParam": "Audio-Reactive Envelope", # New

    # Visualization Nodes
    "AudioDisplayWaveform": "Display Waveform",
    "AudioCompareWaveforms": "Compare Waveforms",
    "AudioShowInfo": "Show Audio Info",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("----------------------------------")
print("### ComfyUI Audio Tools        ###")
print("### Loaded successfully!       ###")
print("----------------------------------")