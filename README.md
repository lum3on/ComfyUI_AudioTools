# ComfyUI Audio Toolkit (AudioTools)

Welcome to the ComfyUI Audio Toolkit (AudioTools), a comprehensive custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that brings a full suite of audio generation, processing, and analysis capabilities into your generative workflows.

This toolkit is designed for a wide range of audio tasks, from podcast enhancement and text-to-speech to creative music manipulation and fully automated, batch-processed audio-reactive visual generation.

-   **Batch Processing**: Most processing, effects, and visualization nodes are now **batch-aware**. You can load an entire folder of audio files, process them all simultaneously, and visualize the results in one go.

## Features

-   **Audio Generation**: Synthesize speech from text using your system's installed voices.
-   **Utility**: Concatenate, pan, pad with silence, and apply fades.
-   **Professional Processing**: Trim, Normalize, Amplify, Mix, Compress, apply EQ, and use a Noise Gate. Most are batch-compatible.
-   **Audio Repair**: Remove silence, de-ess harsh vocals, and de-hum electrical noise.
-   **Creative Effects**: Add Reverb, Delay/Echo, and perform Pitch Shifting or Time Stretching.
-   **AI-Powered Tools**:
    -   **Music Separation**: Split songs into `vocals`, `bass`, `drums`, and `other` stems.
    -   **Voice Enhancement**: Isolate vocals to remove background noise from speech recordings.
    -   **Transcription**: Transcribe audio to text with state-of-the-art accuracy using OpenAI's Whisper.
-   **Advanced I/O**:
    -   **Load an entire folder** of audio files into a single batch.
    -   Iterate through batches using a dedicated list output.
-   **Analysis & Visualization**:
    -   Generate waveform images for single clips or entire batches.
    -   Inspect technical info, measure industry-standard loudness (LUFS), and detect BPM.
-   **Audio-Reactive Generation**: Generate frame-by-frame values from an audio's volume or beat, ready to be plugged into any animation parameter (e.g., zoom, denoise, motion).

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/njlent/ComfyUI_AudioTools
    ```
3.  Install the required dependencies from within the new directory:
    ```bash
    pip install -r ComfyUI_AudioTools/requirements.txt
    ```
4.  Restart ComfyUI.

## Node Reference

### I/O & Batch Nodes
-   **Load Audio Batch (Path)**: Loads all audio files from a folder path that match a pattern (e.g., `*.wav`). Outputs a padded batch *and* an unpadded list for looping.
-   **Get Audio From List**: Retrieves a single audio clip from the `audio_list` output of the batch loader, allowing for individual processing.

### Generation Nodes
-   **Text to Speech**: Converts text into spoken audio.

### Utility Nodes
-   **Concatenate Audio**: Joins two audio clips together end-to-end.
-   **Stereo Panner**: Positions a sound in the stereo (left/right) field.
-   **Pad With Silence**: Adds a specified duration of silence to the beginning or end of audio.

### Processing & Repair Nodes
-   **Amplify / Gain**: Adjusts volume by a dB value.
-   **Normalize Audio**: Normalizes peak volume to a target dB level.
-   **Mix Audio Tracks**: Combines two audio tracks.
-   **Trim Audio**: Cuts seconds from the beginning or end of a clip. (no batches)
-   **Remove Silence**: Intelligently trims silent sections. (no batches)
-   **Noise Gate**: Silences audio that falls below a volume threshold.
-   **De-Esser**: Reduces harsh "s" sounds in voice recordings.
-   **De-Plosive (Low Cut)**: Reduces low-frequency pops.
-   **De-Hum (50/60Hz)**: Removes electrical power line hum.
-   **Parametric EQ for Voice**: A 3-band EQ for enhancing vocal clarity.
-   **Vocal Compressor**: Evens out dynamic range.

### Effects Nodes
-   **Reverb**: Adds spatial reverberation.
-   **Delay / Echo**: Creates a delay/echo effect.
-   **Fade In**: Applies a linear fade-in to the start of the audio.
-   **Fade Out**: Applies a linear fade-out to the end of the audio.
-   **Pitch Shift / Time Stretch**: Changes audio pitch and/or speed. (no batches)

### AI Nodes
-   **Stem Separator (AI)**: Splits music into `vocals`, `bass`, `drums`, and `other`.
-   **Speech Denoise (AI)**: Isolates vocals to remove background noise.
-   **Speech-to-Text (Whisper)**: Transcribes audio to text.

### Analysis & Reactive Nodes
-   **Loudness Meter (LUFS)**: Measures perceived loudness to the EBU R 128 broadcast standard.
-   **BPM Detector / Reactive**: Estimates tempo and outputs a list of `1.0`s on beat frames and `0.0`s otherwise, synced to a target `fps`.
-   **Audio-Reactive Envelope**: Outputs the volume envelope of an audio clip as a frame-by-frame list of floats (0-1), synced to a target `fps`.
-   **Show Audio Info**: Displays technical details (sample rate, duration, batch size, etc.).

### Visualization Nodes
-   **Display Waveform**: Generates an image of the audio's waveform.
-   **Compare Waveforms**: Creates an overlay image of two waveforms.

## Dependencies

-   `torch` & `torchaudio`
-   `demucs`
-   `openai-whisper`
-   `pyttsx3`
-   `matplotlib`
-   `librosa` (for advanced analysis)
-   `pyloudnorm` (for LUFS measurement)