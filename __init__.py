from .processing_nodes import (AmplifyGain, NormalizeAudio, MixAudio, RemoveSilence, 
                               DeEsser, DePlosive, ParametricEQ, VocalCompressor, 
                               Reverb, Delay, PitchTime)
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
    "AudioRemoveSilence": RemoveSilence,
    "AudioDeEsser": DeEsser,
    "AudioDePlosive": DePlosive,
    "AudioParametricEQ": ParametricEQ,
    "AudioVocalCompressor": VocalCompressor,
    
    # Effects Nodes
    "AudioReverb": Reverb,
    "AudioDelay": Delay,
    "AudioPitchTime": PitchTime,

    # AI Nodes
    "AudioStemSeparate": StemSeparator,
    "AudioSpeechDenoise": SpeechDenoise,
    "AudioSpeechToTextWhisper": SpeechToTextWhisper,

    # Visualization Nodes
    "AudioDisplayWaveform": DisplayWaveform,
    "AudioCompareWaveforms": CompareWaveforms,
    "AudioShowInfo": ShowAudioInfo, # New node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Generation Nodes
    "TextToSpeech": "Text to Speech",

    # Processing Nodes
    "AudioAmplify": "Amplify / Gain",
    "AudioNormalize": "Normalize Audio",
    "AudioMix": "Mix Audio Tracks",
    "AudioRemoveSilence": "Remove Silence",
    "AudioDeEsser": "De-Esser",
    "AudioDePlosive": "De-Plosive (Low Cut)",
    "AudioParametricEQ": "Parametric EQ for Voice",
    "AudioVocalCompressor": "Vocal Compressor",
    
    # Effects Nodes
    "AudioReverb": "Reverb",
    "AudioDelay": "Delay / Echo",
    "AudioPitchTime": "Pitch Shift / Time Stretch",

    # AI Nodes
    "AudioStemSeparate": "Stem Separator (AI)",
    "AudioSpeechDenoise": "Speech Denoise (AI)",
    "AudioSpeechToTextWhisper": "Speech-to-Text (Whisper)",

    # Visualization Nodes
    "AudioDisplayWaveform": "Display Waveform",
    "AudioCompareWaveforms": "Compare Waveforms",
    "AudioShowInfo": "Show Audio Info", # New node
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("----------------------------------")
print("### ComfyUI Audio Tools        ###")
print("### Loaded successfully!       ###")
print("----------------------------------")