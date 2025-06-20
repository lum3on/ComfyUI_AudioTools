from .processing_nodes import (AmplifyGain, NormalizeAudio, MixAudio, RemoveSilence, 
                               DeEsser, DePlosive, ParametricEQ, VocalCompressor, 
                               Reverb, Delay, PitchTime, TrimAudio,
                               ConcatenateAudio, StereoPanner, DeHum, NoiseGate,
                               LoudnessMeter, BPMDetector, AudioReactiveParameter,
                               FadeIn, FadeOut, PadAudio)
from .ai_nodes import StemSeparator, SpeechDenoise, SpeechToTextWhisper
from .visualization_nodes import DisplayWaveform, CompareWaveforms, ShowAudioInfo # New import
from .generation_nodes import TextToSpeechNode
from .io_nodes import LoadAudioBatch

NODE_CLASS_MAPPINGS = {
    # Generation Nodes
    "TextToSpeech": TextToSpeechNode,

    # Processing Nodes
    "AudioAmplify": AmplifyGain,
    "AudioNormalize": NormalizeAudio,
    "AudioRemoveSilence": RemoveSilence,
    "AudioDeEsser": DeEsser,
    "AudioDePlosive": DePlosive,
    "AudioParametricEQ": ParametricEQ,
    "AudioVocalCompressor": VocalCompressor,
    "AudioDeHum": DeHum,
    "AudioNoiseGate": NoiseGate,
    
    # Effects Nodes
    "AudioReverb": Reverb,
    "AudioDelay": Delay,
    "AudioPitchTime": PitchTime,
    "AudioFadeIn": FadeIn,
    "AudioFadeOut": FadeOut,

    # Utility Nodes
    "AudioMix": MixAudio,
    "AudioTrim": TrimAudio,
    "AudioConcatenate": ConcatenateAudio,
    "AudioStereoPan": StereoPanner,
    "AudioPad": PadAudio,

    # AI Nodes
    "AudioStemSeparate": StemSeparator,
    "AudioSpeechDenoise": SpeechDenoise,
    "AudioSpeechToTextWhisper": SpeechToTextWhisper,

    # Analysis & Reactive Nodes
    "AudioLoudnessMeter": LoudnessMeter,
    "AudioBPMDetector": BPMDetector,
    "AudioReactiveParam": AudioReactiveParameter,

    # Visualization Nodes
    "AudioDisplayWaveform": DisplayWaveform,
    "AudioCompareWaveforms": CompareWaveforms,
    "AudioShowInfo": ShowAudioInfo,

    # IO Nodes
    "AudioLoadBatch": LoadAudioBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Generation Nodes
    "TextToSpeech": "Text to Speech",

    # Processing Nodes
    "AudioAmplify": "Amplify / Gain",
    "AudioNormalize": "Normalize Audio",
    "AudioRemoveSilence": "Remove Silence",
    "AudioDeEsser": "De-Esser",
    "AudioDePlosive": "De-Plosive (Low Cut)",
    "AudioParametricEQ": "Parametric EQ for Voice",
    "AudioVocalCompressor": "Vocal Compressor",
    "AudioDeHum": "De-Hum (50/60Hz)",
    "AudioNoiseGate": "Noise Gate",
    
    # Effects Nodes
    "AudioReverb": "Reverb",
    "AudioDelay": "Delay / Echo",
    "AudioPitchTime": "Pitch Shift / Time Stretch",
    "AudioFadeIn": "Fade In",
    "AudioFadeOut": "Fade Out", 

    # Utility Nodes
    "AudioMix": "Mix Audio Tracks",
    "AudioTrim": "Trim Audio",
    "AudioConcatenate": "Concatenate Audio",
    "AudioStereoPan": "Stereo Panner",
    "AudioPad": "Pad With Silence",

    # AI Nodes
    "AudioStemSeparate": "Stem Separator (AI)",
    "AudioSpeechDenoise": "Speech Denoise (AI)",
    "AudioSpeechToTextWhisper": "Speech-to-Text (Whisper)",

    # Analysis & Reactive Nodes
    "AudioLoudnessMeter": "Loudness Meter (LUFS)",
    "AudioBPMDetector": "BPM Detector / Reactive",
    "AudioReactiveParam": "Audio-Reactive Envelope",

    # Visualization Nodes
    "AudioDisplayWaveform": "Display Waveform",
    "AudioCompareWaveforms": "Compare Waveforms",
    "AudioShowInfo": "Show Audio Info",

    # IO Nodes
    "AudioLoadBatch": "Load Audio Batch (Path)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("----------------------------------")
print("### ComfyUI Audio Tools        ###")
print("### Loaded successfully!       ###")
print("----------------------------------")