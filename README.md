# ComfyAudio Toolkit

Welcome to the ComfyAudio Toolkit, a custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that brings powerful audio processing capabilities into your generative workflows.

This toolkit is designed for both voice enhancement and creative music manipulation. All nodes are chainable, using a custom `AUDIO` data type that passes both the waveform and sample rate between nodes.

## Features

- **Load, Preview, and Save** common audio formats (WAV, MP3).
- **Essential Processing**: Normalize, Amplify, Trim, and Mix audio.
- **AI-Powered Voice Enhancement**: Remove background noise from speech.
- **AI-Powered Music Tools**: Separate songs into `vocals`, `bass`, `drums`, and `other` stems using Demucs.
- **AI-Powered Transcription**: Transcribe audio to text using OpenAI's Whisper.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/ComfyAudio.git
    ```
    (Replace with your actual repo URL once you create it)
3.  Install the required dependencies:
    ```bash
    pip install -r ComfyAudio/requirements.txt
    ```
4.  Restart ComfyUI.

## Example Workflow: Podcast Cleanup

Here is a simple workflow to clean up a voice recording:

1.  **Load Audio**: Load your raw recording.
2.  **Speech Denoise (AI)**: Connect the audio to remove background noise.
3.  **Normalize Audio**: Connect the denoised audio and set a target peak level (e.g., -1.0 dB).
4.  **Preview Audio**: Listen to the final, cleaned-up result.
5.  **Save Audio**: Save the processed file as a WAV or MP3.

## Node Reference

-   **Load Audio**: Loads an audio file.
-   **Save Audio**: Saves an audio file.
-   **Preview Audio**: Creates a temporary audio file for previewing in the UI.
-   **Amplify/Gain**: Adjusts the volume.
-   **Normalize Audio**: Normalizes the audio to a specific dB level.
-   **Mix Audio Tracks**: Combines two audio tracks.
-   **Stem Separator (AI)**: Splits music into vocals, drums, bass, etc. (Note: This is computationally intensive and may take time).
-   **Speech Denoise (AI)**: Cleans background noise from voice recordings.
-   **Speech-to-Text (Whisper)**: Transcribes audio into text.

## Dependencies

-   `torch` & `torchaudio`
-   `demucs`
-   `openai-whisper`

---
Made with ❤️ for the ComfyUI community.