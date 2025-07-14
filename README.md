# ComfyUI Audio Toolkit (AudioTools)

Welcome to the ComfyUI Audio Toolkit (AudioTools), a comprehensive custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that brings a full suite of audio generation, processing, and analysis capabilities into your generative workflows.

This toolkit is designed for a wide range of audio tasks, from podcast enhancement and text-to-speech to creative music manipulation and fully automated, batch-processed audio-reactive visual generation.

## Node Glossary

### I/O & Batch Nodes
*   [Load Audio Batch (Path)](#📂-load-audio-batch-path)
*   [Get Audio From List](#📂-get-audio-from-list)
*   [Standardize Audio (Format/Channels)](#📂-standardize-audio-formatchannels)

### Generation Nodes
*   [Text to Speech](#💬-text-to-speech)

### Utility Nodes
*   [Concatenate Audio](#-concatenate-audio)
*   [Stereo Panner](#-stereo-panner)
*   [Pad With Silence](#-pad-with-silence)

### Processing & Repair Nodes
*   [Amplify / Gain](#-amplify--gain)
*   [Normalize Audio](#-normalize-audio)
*   [Mix Audio Tracks](#-mix-audio-tracks)
*   [Trim Audio](#-trim-audio)
*   [Remove Silence](#-remove-silence)
*   [Noise Gate](#-noise-gate)
*   [De-Esser](#-de-esser)
*   [De-Plosive (Low Cut)](#-de-plosive-low-cut)
*   [De-Hum (50/60Hz)](#-de-hum-5060hz)
*   [Parametric EQ for Voice](#-parametric-eq-for-voice)
*   [Vocal Compressor](#-vocal-compressor)

### Effects Nodes
*   [Reverb](#-reverb)
*   [Delay / Echo](#-delay--echo)
*   [Fade In](#-fade-in)
*   [Fade Out](#-fade-out)
*   [Pitch Shift / Time Stretch](#-pitch-shift--time-stretch)

### AI Nodes
*   [Stem Separator (AI)](#-stem-separator-ai)
*   [Speech Denoise (AI)](#-speech-denoise-ai)
*   [Speech-to-Text + SRT (Whisper)](#-speech-to-text--srt-whisper)

### Analysis & Reactive Nodes
*   [Loudness Meter (LUFS)](#-loudness-meter-lufs)
*   [BPM Detector / Reactive](#-bpm-detector--reactive)
*   [Audio-Reactive Envelope](#-audio-reactive-envelope)
*   [Show Audio Info](#-show-audio-info)

### Visualization Nodes
*   [Display Waveform](#-display-waveform)
*   [Compare Waveforms](#-compare-waveforms)

---

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/lum3on/ComfyUI_AudioTools
    ```
3.  Install the required dependencies from within the new directory:
    ```bash
    pip install -r ComfyUI_AudioTools/requirements.txt
    ```
4.  Restart ComfyUI.

-   **Batch Processing**: Most processing, effects, and visualization nodes are now **batch-aware**. You can load an entire folder of audio files, process them all simultaneously, and visualize the results in one go.

---

## Node Reference

### I/O & Batch Nodes

#### 📂 Load Audio Batch (Path)
*Category: `AudioTools/IO`*

![Load Audio Batch (Path)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Load_Audio_Batch_Path.jpg?raw=true)

Loads all audio files from a folder path that match a pattern (e.g., `*.wav`). Can be configured to sort files in various ways, including by modification date to get the newest file.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `directory_path` | `STRING` | The full path to the directory containing audio files. |
| **(Input)** `file_pattern` | `STRING` | The pattern to match files (e.g., `*.wav`, `*.mp3`, `audio_*.flac`). |
| **(Input)** `sort_order` | `COMBO` | The order to sort files before loading. |
| **(Input)** `skip_first` | `INT` | Number of files to skip from the start of the sorted list. |
| **(Input)** `load_cap` | `INT` | Maximum number of files to load. Use -1 for no limit. |
| **(Output)** `audio_batch` | `AUDIO` | A single padded `AUDIO` object containing all loaded clips as a batch. |
| **(Output)** `audio_list` | `AUDIO_LIST` | A list where each item is a separate, unpadded `AUDIO` clip. |
| **(Output)** `filenames` | `STRING` | A list of the filenames (string) that were successfully loaded. |

#### 📂 Get Audio From List
*Category: `AudioTools/IO`*

![Get Audio From List](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Get_Audio_From_List.jpg?raw=true)

Retrieves a single audio clip from the `audio_list` output of the batch loader, allowing for individual processing within a workflow.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio_list` | `AUDIO_LIST` | The list of audio clips from a batch loader. |
| **(Input)** `index` | `INT` | The index of the audio clip to retrieve from the list. Wraps around if the index is out of bounds. |
| **(Output)** `AUDIO` | `AUDIO` | The single audio clip selected from the list. |

#### 📂 Standardize Audio (Format/Channels)
*Category: `AudioTools/Processing`*

![Standardize Audio (Format/Channels)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Standardize_Audio_Format_Channels.jpg?raw=true)

Converts audio to a standard format (mono or stereo) and data type to fix compatibility issues with other nodes that expect a specific layout.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio (or batch) to standardize. |
| **(Input)** `channel_layout` | `COMBO` | Convert audio to mono (single channel) or ensure it is stereo (two channels). |
| **(Output)** `AUDIO` | `AUDIO` | The standardized audio. |

### Generation Nodes

#### 💬 Text to Speech
*Category: `AudioTools/Generation`*

![Text to Speech](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Text_to_Speech.jpg?raw=true)

Converts text into spoken audio using your operating system's built-in text-to-speech engine.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `text` | `STRING` | The text to be converted into speech. |
| **(Input)** `voice_index` | `COMBO` | The system voice to use for synthesis. Voice options depend on your operating system. |
| **(Input)** `rate` | `INT` | The speaking rate in words per minute. |
| **(Input)** `volume` | `FLOAT` | The volume of the generated audio (0.0 to 1.0). |
| **(Output)** `AUDIO` | `AUDIO` | The generated spoken audio clip. |

### Utility Nodes

#### 🛠️ Concatenate Audio
*Category: `AudioTools/Utility`*

Joins two audio clips together end-to-end. The first clip from each input batch is used.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio_a` | `AUDIO` | The first audio clip (will be the beginning of the result). |
| **(Input)** `audio_b` | `AUDIO` | The second audio clip (will be appended to the end of the first). |
| **(Output)** `AUDIO` | `AUDIO` | The new, combined audio clip. |

#### 🛠️ Stereo Panner
*Category: `AudioTools/Utility`*

Positions a sound in the stereo (left/right) field. This is applied to all audio clips in a batch.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to pan. |
| **(Input)** `pan` | `FLOAT` | Stereo position. -1.0 is hard left, 1.0 is hard right, 0.0 is center. |
| **(Output)** `AUDIO` | `AUDIO` | The panned audio. Mono inputs are converted to stereo. |

#### 🛠️ Pad With Silence
*Category: `AudioTools/Utility`*

Adds a specified duration of silence to the beginning or end of audio. This is applied to all audio clips in a batch.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to pad with silence. |
| **(Input)** `pad_start_seconds` | `FLOAT` | Duration of silence to add to the beginning of the audio. |
| **(Input)** `pad_end_seconds` | `FLOAT` | Duration of silence to add to the end of the audio. |
| **(Output)** `AUDIO` | `AUDIO` | The padded audio. |

### Processing & Repair Nodes

#### 🔧 Amplify / Gain
*Category: `AudioTools/Processing`*

![Amplify / Gain](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Amplify___Gain.jpg?raw=true)

Adjusts the volume of the audio by a specified decibel (dB) value.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to apply gain to. |
| **(Input)** `gain_db` | `FLOAT` | Amount of gain in decibels (dB) to apply. Positive values amplify, negative values attenuate. |
| **(Output)** `AUDIO` | `AUDIO` | The amplified audio. |

#### 🔧 Normalize Audio
*Category: `AudioTools/Processing`*

![Normalize Audio](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Normalize_Audio.jpg?raw=true)

Normalizes the peak volume of the audio to a target dB level, maximizing loudness without clipping.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to normalize. |
| **(Input)** `target_level_db` | `FLOAT` | The target peak volume in decibels (dB). A value of 0.0 is maximum, but -1.0 is a common target to avoid clipping. |
| **(Output)** `AUDIO` | `AUDIO` | The normalized audio. |

#### 🔧 Mix Audio Tracks
*Category: `AudioTools/Processing`*

![Mix Audio Tracks](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Mix_Audio_Tracks.jpg?raw=true)

Combines two audio tracks into one. The first clip from each input batch is used.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio_1` | `AUDIO` | The first audio track. |
| **(Input)** `audio_2` | `AUDIO` | The second audio track. |
| **(Input)** `gain_1_db` | `FLOAT` | Gain in dB for the first audio track. |
| **(Input)** `gain_2_db` | `FLOAT` | Gain in dB for the second audio track. |
| **(Output)** `AUDIO` | `AUDIO` | The mixed audio. |

#### 🔧 Trim Audio
*Category: `AudioTools/Processing`*

![Trim Audio](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Trim_Audio.jpg?raw=true)

Cuts a specified number of seconds from the beginning or end of an audio clip. (Note: This node is not batch-aware).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to trim. |
| **(Input)** `trim_start_seconds` | `FLOAT` | Number of seconds to cut from the beginning of the audio. |
| **(Input)** `trim_end_seconds` | `FLOAT` | Number of seconds to cut from the end of the audio. |
| **(Output)** `AUDIO` | `AUDIO` | The trimmed audio clip. |

#### 🔧 Remove Silence
*Category: `AudioTools/Processing`*

![Remove Silence](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Remove_Silence.jpg?raw=true)

Intelligently analyzes and trims silent sections from an audio clip. (Note: This node is not batch-aware).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip from which to remove silent sections. |
| **(Input)** `silence_threshold_db` | `FLOAT` | The volume level (in dB) below which audio is considered silent. |
| **(Input)** `min_silence_len_ms` | `INT` | The minimum duration (in milliseconds) of silence to be removed. |
| **(Output)** `AUDIO` | `AUDIO` | The audio with silent sections removed. |

#### 🔧 Noise Gate
*Category: `AudioTools/Processing`*

![Noise Gate](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Noise_Gate.jpg?raw=true)

Silences audio that falls below a specified volume threshold, useful for removing background noise between words.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to apply the noise gate to. |
| **(Input)** `threshold_db` | `FLOAT` | The volume level (dB) below which the gate will close and silence the audio. |
| **(Input)** `attack_ms` | `FLOAT` | How quickly (in ms) the gate opens when the signal exceeds the threshold. |
| **(Input)** `release_ms` | `FLOAT` | How quickly (in ms) the gate closes after the signal falls below the threshold. |
| **(Output)** `AUDIO` | `AUDIO` | The gated audio. |

#### 🔧 De-Esser
*Category: `AudioTools/Processing`*

![De-Esser](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/De-Esser.jpg?raw=true)

Reduces harsh "s" sounds (sibilance) in voice recordings by applying a narrow-band EQ cut at a specified frequency.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to de-ess. |
| **(Input)** `frequency_hz` | `INT` | The center frequency of sibilance to target (typically 5-8 kHz). |
| **(Input)** `reduction_db` | `FLOAT` | The amount of gain reduction (in dB) to apply at the target frequency. |
| **(Input)** `q_factor` | `FLOAT` | The width of the frequency band to affect. Higher Q is narrower. |
| **(Output)** `AUDIO` | `AUDIO` | The de-essed audio. |

#### 🔧 De-Plosive (Low Cut)
*Category: `AudioTools/Processing`*

![De-Plosive (Low Cut)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/De-Plosive_Low_Cut.jpg?raw=true)

Reduces low-frequency pops ("plosives") caused by air hitting the microphone (e.g., from 'p' and 'b' sounds) using a high-pass filter.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to de-plosive (low-cut). |
| **(Input)** `cutoff_hz` | `INT` | The cutoff frequency for the high-pass filter. Frequencies below this will be rolled off. |
| **(Output)** `AUDIO` | `AUDIO` | The filtered audio. |

#### 🔧 De-Hum (50/60Hz)
*Category: `AudioTools/Processing`*

![De-Hum (50/60Hz)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/De-Hum_50_60Hz.jpg?raw=true)

Removes electrical power line hum by applying very narrow notch filters at the fundamental frequency (50 or 60 Hz) and its first harmonic.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to de-hum. |
| **(Input)** `hum_freq` | `COMBO` | The fundamental frequency of the electrical hum to remove. |
| **(Input)** `reduction_db` | `FLOAT` | The amount of gain reduction (dB) to apply to the hum frequencies. |
| **(Input)** `q_factor` | `FLOAT` | The narrowness of the filter. A high Q value is needed to target only the hum. |
| **(Output)** `AUDIO` | `AUDIO` | The de-hummed audio. |

#### 🔧 Parametric EQ for Voice
*Category: `AudioTools/Processing`*

![Parametric EQ for Voice](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Parametric_EQ_for_Voice.jpg?raw=true)

A 3-band equalizer specifically tuned for enhancing vocal clarity, featuring a low-cut, a presence boost, and an "air" band.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to equalize. |
| **(Input)** `low_cut_hz` | `INT` | Low-cut (high-pass) filter to remove rumble. 80-120Hz is common for voice. |
| **(Input)** `presence_boost_db` | `FLOAT` | Boost/cut for vocal presence (around 4kHz). |
| **(Input)** `air_boost_db` | `FLOAT` | High-shelf boost/cut for 'air' and clarity (around 12kHz). |
| **(Output)** `AUDIO` | `AUDIO` | The equalized audio. |

#### 🔧 Vocal Compressor
*Category: `AudioTools/Processing`*

![Vocal Compressor](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Vocal_Compressor.jpg?raw=true)

Evens out the dynamic range of an audio clip, making quiet parts louder and loud parts quieter for a more consistent volume level.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to compress. |
| **(Input)** `threshold_db` | `FLOAT` | The volume level (dB) at which the compressor starts working. |
| **(Input)** `ratio` | `FLOAT` | The amount of gain reduction (e.g., 4.0 means a 4:1 ratio). |
| **(Input)** `attack_ms` | `FLOAT` | How quickly (in ms) the compressor reacts to loud sounds. |
| **(Input)** `release_ms` | `FLOAT` | How quickly (in ms) the compressor stops after the sound falls below the threshold. |
| **(Input)** `makeup_gain_db` | `FLOAT` | Volume boost to apply after compression to make up for the reduced level. |
| **(Output)** `AUDIO` | `AUDIO` | The compressed audio. |

### Effects Nodes

#### ✨ Reverb
*Category: `AudioTools/Effects`*

Adds spatial reverberation to the audio, simulating the sound of a room or space.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to apply reverb to. |
| **(Input)** `room_size` | `FLOAT` | The perceived size of the reverberant space (0-100). |
| **(Input)** `damping` | `FLOAT` | How much the high frequencies are absorbed in the reverb tails (0-100). |
| **(Input)** `wet_level` | `FLOAT` | The volume of the reverberated (wet) signal. |
| **(Input)** `dry_level` | `FLOAT` | The volume of the original (dry) signal. |
| **(Output)** `AUDIO` | `AUDIO` | The reverberated audio. |

#### ✨ Delay / Echo
*Category: `AudioTools/Effects`*

Creates a repeating, decaying echo effect on the audio.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to apply delay to. |
| **(Input)** `delay_ms` | `FLOAT` | The time (in milliseconds) between each echo. |
| **(Input)** `feedback` | `FLOAT` | How much of the delayed signal is fed back into the delay line, creating more echoes. |
| **(Input)** `mix` | `FLOAT` | The balance between the original (dry) and delayed (wet) signal. 0.0 is all dry, 1.0 is all wet. |
| **(Output)** `AUDIO` | `AUDIO` | The audio with the delay effect. |

#### ✨ Fade In
*Category: `AudioTools/Effects`*

Applies a linear fade-in from silence to the start of the audio.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to apply a fade-in to. |
| **(Input)** `duration_seconds` | `FLOAT` | The duration of the fade-in effect in seconds. |
| **(Output)** `AUDIO` | `AUDIO` | The audio with the fade-in applied. |

#### ✨ Fade Out
*Category: `AudioTools/Effects`*

Applies a linear fade-out to silence at the end of the audio.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to apply a fade-out to. |
| **(Input)** `duration_seconds` | `FLOAT` | The duration of the fade-out effect in seconds. |
| **(Output)** `AUDIO` | `AUDIO` | The audio with the fade-out applied. |

#### ✨ Pitch Shift / Time Stretch
*Category: `AudioTools/Effects`*

Changes the audio's pitch without changing its speed, and/or changes its speed without changing the pitch. (Note: This node is not batch-aware).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio to pitch shift or time stretch. |
| **(Input)** `pitch_semitones` | `FLOAT` | The number of semitones to shift the pitch up or down. |
| **(Input)** `tempo_factor` | `FLOAT` | The factor by which to change the tempo. >1.0 is faster, <1.0 is slower. |
| **(Output)** `AUDIO` | `AUDIO` | The processed audio. |

### AI Nodes

#### 🧠 Stem Separator (AI)
*Category: `AudioTools/AI`*

![Stem Separator (AI)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Stem_Separator_AI.jpg?raw=true)

Uses the Demucs AI model to split a music track into its core components: vocals, bass, drums, and other instruments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to be separated into stems. |
| **(Input)** `model_name` | `COMBO` | The Demucs model to use for separation. 'htdemucs_ft' is a good general-purpose choice. |
| **(Output)** `vocals` | `AUDIO` | The isolated vocal track. |
| **(Output)** `bass` | `AUDIO` | The isolated bass track. |
| **(Output)** `drums` | `AUDIO` | The isolated drum track. |
| **(Output)** `other` | `AUDIO` | All other musical elements combined. |

#### 🧠 Speech Denoise (AI)
*Category: `AudioTools/AI`*

![Speech Denoise (AI)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Speech_Denoise_AI.jpg?raw=true)

Uses the Demucs AI model to isolate vocals from a recording, effectively removing background noise and non-vocal sounds.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip containing speech to be denoised. |
| **(Input)** `model_name` | `COMBO` | The Demucs model to use for isolating vocals. It will remove non-vocal sounds. |
| **(Output)** `AUDIO` | `AUDIO` | The denoised audio containing only the vocal signal. |

#### 🧠 Speech-to-Text + SRT (Whisper)
*Category: `AudioTools/AI`*

![Speech-to-Text + SRT (Whisper)](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Speech-to-Text_SRT_Whisper.jpg?raw=true)

Transcribes audio to text using OpenAI's Whisper model. Can optionally generate a timed SRT (SubRip Subtitle) formatted string.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to be transcribed. |
| **(Input)** `model_size` | `COMBO` | The size of the Whisper model to use. Larger models are more accurate but slower. |
| **(Input)** `language` | `COMBO` | The language of the speech in the audio. 'Auto-Detect' is an option. |
| **(Input)** `task` | `COMBO` | Choose between standard transcription or translating the speech directly to English. |
| **(Input)** `generate_srt` | `BOOLEAN` | If enabled, generates a timestamped SRT subtitle string in the `srt_text` output. |
| **(Input)** `srt_max_line_len`| `INT` | (Optional) The maximum number of characters allowed per line in an SRT block. |
| **(Input)** `srt_max_lines`| `INT` | (Optional) The maximum number of lines allowed per SRT block. |
| **(Input)** `srt_max_duration_sec`| `FLOAT` | (Optional) The maximum duration in seconds an SRT block can cover. |
| **(Output)** `text` | `STRING` | The transcribed text as a single string. |
| **(Output)** `srt_text` | `STRING` | The generated SRT subtitle string with timestamps. Empty if `generate_srt` is disabled. |

### Analysis & Reactive Nodes

#### 📈 Loudness Meter (LUFS)
*Category: `AudioTools/Analysis`*

Measures the perceived loudness of an audio clip according to the EBU R 128 broadcast standard and outputs the result as a string.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to measure for loudness. |
| **(Output)** `loudness_info` | `STRING` | A text string containing the measured integrated loudness in LUFS. |

#### 📈 BPM Detector / Reactive
*Category: `AudioTools/Analysis`*

Estimates the tempo (Beats Per Minute) of an audio clip and generates a frame-synced list of beat events.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to detect the tempo from. |
| **(Input)** `fps` | `INT` | The target frames per second to sync the beat events list with. |
| **(Output)** `bpm_info` | `STRING` | A text string with the estimated BPM. |
| **(Output)** `beat_events` | `FLOAT` | A list of floats, with `1.0` on frames that land on a beat and `0.0` otherwise. |

#### 📈 Audio-Reactive Envelope
*Category: `AudioTools/Analysis`*

Analyzes the volume envelope (RMS) of an audio clip and outputs it as a frame-by-frame list of floats, perfect for driving animations.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to analyze for its volume envelope. |
| **(Input)** `fps` | `INT` | The target frames per second to sync the envelope list with. |
| **(Input)** `smoothing` | `FLOAT` | Amount of smoothing to apply to the envelope. 0 is no smoothing, 1 is maximum smoothing. |
| **(Output)** `envelope` | `FLOAT` | A list of floats (normalized 0-1) representing the audio's volume for each frame. |

#### 📈 Show Audio Info
*Category: `AudioTools/Visualization`*

![Show Audio Info](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Show_Audio_Info.jpg?raw=true)

An output node that displays technical details about an audio clip or batch, such as sample rate, duration, batch size, and more.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip to display information about. |
| **(Output)** `info` | `STRING` | A text string containing the technical details of the audio. |

### Visualization Nodes

#### 📊 Display Waveform
*Category: `AudioTools/Visualization`*

![Display Waveform](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Display_Waveform.jpg?raw=true)

Generates and displays an image of the audio's waveform. This node is batch-aware and will produce one image for each audio clip in the batch.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio` | `AUDIO` | The audio clip(s) to visualize. |
| **(Input)** `width` | `INT` | The width of the output image in pixels. |
| **(Input)** `height` | `INT` | The height of the output image in pixels. |
| **(Input)** `line_color` | `STRING` | The color of the waveform line (hex code). |
| **(Input)** `bg_color` | `STRING` | The background color of the image (hex code). |
| **(Input)** `show_axis` | `BOOLEAN` | Whether to display the time and amplitude axes. |
| **(Output)** `IMAGE` | `IMAGE` | An image (or batch of images) of the waveform. |

#### 📊 Compare Waveforms
*Category: `AudioTools/Visualization`*

![Compare Waveforms](https://github.com/lum3on/ComfyUI_AudioTools/blob/main/readme/Compare_Waveforms.jpg?raw=true)

Creates an overlay image of two waveforms, making it easy to see the difference before and after processing. This node is batch-aware.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **(Input)** `audio_a` | `AUDIO` | The first audio clip (or batch) to compare. |
| **(Input)** `audio_b` | `AUDIO` | The second audio clip (or batch) to compare. |
| **(Input)** `width` | `INT` | The width of the output image in pixels. |
| **(Input)** `height` | `INT` | The height of the output image in pixels. |
| **(Input)** `color_a` | `STRING` | The color for Audio A's waveform (hex code). |
| **(Input)** `color_b` | `STRING` | The color for Audio B's waveform (hex code). |
| **(Input)** `bg_color` | `STRING` | The background color of the image (hex code). |
| **(Input)** `show_axis` | `BOOLEAN` | Whether to display the time and amplitude axes and a legend. |
| **(Output)** `IMAGE` | `IMAGE` | An image (or batch of images) comparing the two waveforms. |

## Dependencies

-   `torch` & `torchaudio`
-   `demucs`
-   `openai-whisper`
-   `pyttsx3`
-   `matplotlib`
-   `librosa` (for advanced analysis)
-   `pyloudnorm` (for LUFS measurement)

---
<details>
<summary>Changelog</summary>

Recent Changes:

**Version 1.1.06** 
- fixed compatibility of the Speech-to-text Node with https://github.com/niknah/ComfyUI-F5-TTS
- added "Standardize Audio Format/Channels" Node to fix future compatibility issues

**Version 1.1.05** 
- added srt output to the Speech-to-Text Node
- fixed some edge case errors in the Speech-to-text Node

**Version 1.1.01** 
- added tooltips on hover to almost all in/outputs
- added licence + disclaimer

**Version 1.1.00**
- init

</details>