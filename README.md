# ComfyUI Audio Toolkit

Welcome to the ComfyUI Audio Toolkit, a comprehensive custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that brings a full suite of audio generation, processing, and analysis capabilities into your generative workflows.

This toolkit is designed for a wide range of audio tasks, from podcast enhancement and text-to-speech to creative music manipulation and AI-powered transcription.

## Core Concept: The `AUDIO` Data Type

All nodes in this toolkit are designed to be chainable. They use a custom `AUDIO` data type that bundles the audio waveform (as a tensor) and its sample rate. This ensures seamless compatibility and allows you to build complex processing chains by simply connecting one node to the next.

## Features

-   **I/O**: Load, Preview, and Save common audio formats (WAV, MP3, FLAC).
-   **Audio Generation**: Synthesize speech from text using your system's installed voices.
-   **Essential Processing**: Normalize, Amplify, Mix, Trim silence, Compress, and apply professional EQ.
-   **Creative Effects**: Add Reverb, Delay/Echo, and perform Pitch Shifting or Time Stretching.
-   **AI-Powered Tools**:
    -   **Music Separation**: Split songs into `vocals`, `bass`, `drums`, and `other` stems using Demucs.
    -   **Voice Enhancement**: Isolate vocals to remove background noise from speech recordings.
    -   **Transcription**: Transcribe audio to text with state-of-the-art accuracy using OpenAI's Whisper.
-   **Visualization & Analysis**: Generate waveform images, compare two audio signals visually, and inspect detailed technical information.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/njlent-comfyui_audiotools.git
    ```
    *(Replace with the actual repository URL)*
3.  Install the required dependencies from within the new directory:
    ```bash
    pip install -r njlent-comfyui_audiotools/requirements.txt
    ```
4.  Restart ComfyUI.

## Example Workflows

### 1. Podcast Cleanup & Voice Enhancement

This workflow takes a raw voice recording and makes it sound professional and clean.

1.  **Load Audio**: Load your raw WAV or MP3 recording.
2.  **Speech Denoise (AI)**: Connect the audio to remove background noise.
3.  **Remove Silence**: Automatically trim long pauses from the speech.
4.  **Parametric EQ for Voice**: Apply a low-cut to remove rumble and add a presence/air boost for clarity.
5.  **Vocal Compressor**: Even out the volume levels for a consistent, professional sound.
6.  **Normalize Audio**: Bring the final audio to a standard peak level (e.g., -1.0 dB).
7.  **Preview Audio**: Listen to the final, cleaned-up result.
8.  **Save Audio**: Save the processed file as a new WAV or MP3.

### 2. AI Music Stem Separation

This workflow demonstrates how to isolate the vocals from a song.

1.  **Load Audio**: Load a music track.
2.  **Stem Separator (AI)**: Connect the audio. This node is computationally intensive and will take time to process.
3.  **Connect the `vocals` output** to a **Preview Audio** or **Save Audio** node.
4.  *(Optional)*: Mix the stems back together. For example, connect `bass`, `drums`, and `other` to an **AudioMix** node to create an instrumental version.

## Node Reference

### I/O Nodes
-   **Load Audio**: Loads an audio file (WAV, MP3, etc.) from your ComfyUI `input` directory. Supports file uploads.
-   **Save Audio**: Saves the processed audio to your `output` directory in `wav`, `mp3`, or `flac` format.
-   **Preview Audio**: Creates an interactive audio player in the UI to listen to the audio at any step.

### Generation Nodes
-   **Text to Speech**: Converts a string of text into spoken audio using your system's available TTS voices.

### Processing Nodes
-   **Amplify / Gain**: Adjusts the volume with a specific dB gain.
-   **Normalize Audio**: Normalizes the audio to a target peak dB level.
-   **Mix Audio Tracks**: Combines two audio tracks, with individual gain control for each.
-   **Remove Silence**: Intelligently removes silent sections from an audio clip.
-   **De-Esser**: Reduces harsh "s" sounds (sibilance) in voice recordings.
-   **De-Plosive (Low Cut)**: A high-pass filter to reduce low-frequency pops from "p" and "b" sounds.
-   **Parametric EQ for Voice**: A simple 3-band equalizer designed to enhance vocal clarity (Low-Cut, Presence, Air).
-   **Vocal Compressor**: Evens out the dynamic range, making quiet parts louder and loud parts quieter.

### Effects Nodes
-   **Reverb**: Adds spatial reverberation to the audio.
-   **Delay / Echo**: Creates a delay or echo effect with feedback control.
-   **Pitch Shift / Time Stretch**: Changes the audio's pitch (in semitones) and/or playback speed.

### AI Nodes
-   **Stem Separator (AI)**: Splits a music track into four stems: `vocals`, `bass`, `drums`, and `other`.
-   **Speech Denoise (AI)**: Isolates the vocal stem from an audio track to effectively remove background noise.
-   **Speech-to-Text (Whisper)**: Transcribes audio into text using OpenAI's Whisper. Supports multiple languages and translation to English.

### Visualization Nodes
-   **Display Waveform**: Generates a PNG image of the audio's waveform.
-   **Compare Waveforms**: Creates an overlay image of two different audio waveforms for comparison.
-   **Show Audio Info**: Displays technical details (sample rate, duration, channels, etc.) about the audio signal in a text box.

## Dependencies

This node pack relies on several powerful libraries. The `requirements.txt` file will handle their installation:

-   `torch` & `torchaudio`
-   `demucs` (for AI-powered separation and denoising)
-   `openai-whisper` (for AI-powered transcription)
-   `pyttsx3` (for Text-to-Speech)
-   `matplotlib` (for waveform visualization)

---
Made with ❤️ for the ComfyUI community.