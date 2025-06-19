# File: processing_nodes.py (Final Compatible Version)

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import math

import numpy as np
import librosa
import pyloudnorm as pyln

# --- helper ---

def _manual_reverb(waveform, sample_rate, room_size, damping, wet_level, dry_level):
    """A manual, PyTorch-based implementation of an algorithmic reverb."""
    
    num_channels, num_samples = waveform.shape
    device = waveform.device

    # --- Reverb Parameters based on user input ---
    # Scale room_size (0-100) to a reverb time (0.1-1.0s) and gain (0.5-0.99)
    room_size_factor = room_size / 100.0
    decay_time = 0.1 + room_size_factor * 0.9
    feedback_gain = 0.5 + room_size_factor * 0.49

    # Damping (0-100) controls a simple low-pass filter in the feedback path
    damping_factor = damping / 100.0
    damping_coeff = 1.0 - damping_factor * 0.7 # High damping = more filtering

    # --- Filter Definitions ---
    # Using prime-ish numbers for delay lengths to avoid metallic resonance
    comb_delays = [int(d * sample_rate) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
    allpass_delays = [int(d * sample_rate) for d in [0.005, 0.0017]]
    allpass_gain = 0.7

    # Initialize delay line buffers
    comb_buffers = [torch.zeros(num_channels, d, device=device) for d in comb_delays]
    allpass_buffers = [torch.zeros(num_channels, d, device=device) for d in allpass_delays]
    
    comb_pos = [0] * 4
    allpass_pos = [0] * 2

    output_waveform = torch.zeros_like(waveform)

    # Process sample by sample
    for i in range(num_samples):
        s_in = waveform[:, i]
        s_comb_out = torch.zeros(num_channels, device=device)

        # 1. Parallel Comb Filters to create echoes
        for j in range(len(comb_delays)):
            delayed_sample = comb_buffers[j][:, comb_pos[j]]
            
            # Apply damping (simple low-pass) to the feedback
            damped_sample = delayed_sample * damping_coeff
            
            # Store new value in buffer
            comb_buffers[j][:, comb_pos[j]] = s_in + damped_sample * feedback_gain
            comb_pos[j] = (comb_pos[j] + 1) % comb_delays[j]
            
            s_comb_out += delayed_sample

        # 2. Serial All-Pass Filters to diffuse the sound
        s_allpass_in = s_comb_out
        for j in range(len(allpass_delays)):
            delayed_sample = allpass_buffers[j][:, allpass_pos[j]]
            
            s_allpass_out = -s_allpass_in * allpass_gain + delayed_sample
            allpass_buffers[j][:, allpass_pos[j]] = s_allpass_in + s_allpass_out * allpass_gain
            allpass_pos[j] = (allpass_pos[j] + 1) % allpass_delays[j]
            
            s_allpass_in = s_allpass_out
            
        s_wet = s_allpass_in
        s_dry = waveform[:, i]
        
        output_waveform[:, i] = (s_dry * dry_level + s_wet * wet_level).clamp(-1.0, 1.0)

    return output_waveform


def _calculate_peaking_coeffs(gain_db, q, center_freq, sample_rate):
    A = 10**(gain_db / 40.0)
    w0 = 2.0 * math.pi * center_freq / sample_rate
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return b0/a0, b1/a0, b2/a0, a0/a0, a1/a0, a2/a0


def _calculate_highshelf_coeffs(gain_db, q, cutoff_freq, sample_rate):
    A = 10**(gain_db / 40.0)
    w0 = 2.0 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
    return b0/a0, b1/a0, b2/a0, a0/a0, a1/a0, a2/a0


def _manual_compressor(waveform, sample_rate, threshold_db, ratio, attack_ms, release_ms, makeup_gain_db=0.0):
    """A manual, PyTorch-based implementation of a dynamic range compressor that is multi-channel safe."""
    
    # Convert parameters to useful units
    threshold_lin = 10**(threshold_db / 20.0)
    # Use torch.tensor for coefficients to ensure they are on the correct device
    attack_coeff = torch.tensor(math.exp(-1.0 / (sample_rate * (attack_ms / 1000.0))), device=waveform.device)
    release_coeff = torch.tensor(math.exp(-1.0 / (sample_rate * (release_ms / 1000.0))), device=waveform.device)
    makeup_gain_lin = 10**(makeup_gain_db / 20.0)

    num_channels, num_samples = waveform.shape
    envelope = torch.zeros(num_channels, device=waveform.device)
    gain = torch.ones(num_channels, device=waveform.device)
    output_waveform = torch.zeros_like(waveform)

    # Process sample by sample
    for i in range(num_samples):
        # Peak detection for the envelope - this is a single value across all channels
        input_peak = torch.max(torch.abs(waveform[:, i]))
        
        # --- THE FIX: Use torch.where for conditional logic on tensors ---
        # Instead of a Python 'if', we create a boolean mask and use torch.where.
        # This allows different attack/release behavior for each channel if needed,
        # but here we use a single peak for all channels for simplicity.
        
        # Determine whether to use attack or release coefficient
        # True where input_peak is greater than the channel's envelope
        use_attack = input_peak > envelope
        
        # Apply attack to some channels and release to others using the mask
        envelope = torch.where(
            use_attack,
            attack_coeff * envelope + (1 - attack_coeff) * input_peak, # if True (attack)
            release_coeff * envelope + (1 - release_coeff) * input_peak  # if False (release)
        )
        
        # Gain computation
        # Add a small epsilon to avoid log10(0)
        envelope_db = 20 * torch.log10(envelope + 1e-9)
        
        # Create a mask for channels that are above the threshold
        above_threshold = envelope_db > threshold_db
        
        # Calculate the gain reduction that *would* be applied if above threshold
        gain_reduction_db = (threshold_db - envelope_db) * (1.0 - (1.0 / ratio))
        reduced_gain = 10**(gain_reduction_db / 20.0)
        
        # Use torch.where to apply the reduced_gain only to the channels above threshold
        gain = torch.where(
            above_threshold,
            reduced_gain,           # if True
            1.0                     # if False
        )

        # Apply gain to the current sample for all channels at once
        output_waveform[:, i] = waveform[:, i] * gain * makeup_gain_lin
        
    return output_waveform.clamp(-1.0, 1.0)

# --- nodes ---

class AmplifyGain:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_gain"
    CATEGORY = "L3/AudioTools/Processing"
    def apply_gain(self, audio: dict, gain_db: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        processed_w = F.gain(w, gain_db).clamp(-1.0, 1.0)
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class NormalizeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "target_level_db": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize"
    CATEGORY = "L3/AudioTools/Processing"
    def normalize(self, audio: dict, target_level_db: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        current_peak = 20 * torch.log10(torch.max(torch.abs(w)))
        gain_to_apply = 0 if torch.isinf(current_peak) else target_level_db - current_peak
        processed_w = F.gain(w, gain_to_apply).clamp(-1.0, 1.0)
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class MixAudio:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio_1": ("AUDIO",), "audio_2": ("AUDIO",), "gain_1_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 6.0, "step": 0.1}), "gain_2_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 6.0, "step": 0.1}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "mix"
    CATEGORY = "L3/AudioTools/Utility"
    def mix(self, audio_1: dict, audio_2: dict, gain_1_db: float, gain_2_db: float):
        w1, sr1 = audio_1["waveform"][0], audio_1["sample_rate"]
        w2, sr2 = audio_2["waveform"][0], audio_2["sample_rate"]
        if sr1 != sr2: w2 = F.resample(w2, orig_freq=sr2, new_freq=sr1)
        w1, w2 = F.gain(w1, gain_1_db), F.gain(w2, gain_2_db)
        len1, len2 = w1.shape[1], w2.shape[1]
        if len1 > len2: w2 = torch.nn.functional.pad(w2, (0, len1 - len2))
        elif len2 > len1: w1 = torch.nn.functional.pad(w1, (0, len2 - len1))
        mixed = (w1 + w2).clamp(-1.0, 1.0)
        return ({"waveform": mixed.unsqueeze(0), "sample_rate": sr1},)

class TrimAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "trim_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "How many seconds to cut from the beginning of the audio."}),
                "trim_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "How many seconds to cut from the end of the audio."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "trim"
    CATEGORY = "L3/AudioTools/Utility"
    
    def trim(self, audio: dict, trim_start_seconds: float, trim_end_seconds: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        num_channels, total_samples = w.shape
        
        # Calculate start and end samples to trim to
        start_sample = int(trim_start_seconds * sample_rate)
        end_sample_offset = int(trim_end_seconds * sample_rate)
        end_sample = total_samples - end_sample_offset
        
        # --- Safety Checks ---
        # Ensure start is not negative
        start_sample = max(0, start_sample)
        # Ensure end is not past the start
        end_sample = max(start_sample, end_sample)
        # Ensure end is not beyond the total length
        end_sample = min(total_samples, end_sample)

        # If the trim results in a zero or negative length, return silent audio
        if start_sample >= end_sample:
            print(f"ComfyAudio Warning: Trim result for is empty. Returning silent audio.")
            trimmed_w = torch.zeros((num_channels, 0), device=w.device)
        else:
            # Perform the slice
            trimmed_w = w[:, start_sample:end_sample]
            
        return ({"waveform": trimmed_w.unsqueeze(0), "sample_rate": sample_rate},)

class RemoveSilence:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "silence_threshold_db": ("FLOAT", {"default": -40.0, "min": -90.0, "max": 0.0, "step": 1.0}), "min_silence_len_ms": ("INT", {"default": 500, "min": 50, "max": 5000, "step": 50}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "remove_silence"
    CATEGORY = "L3/AudioTools/Processing"
    def remove_silence(self, audio: dict, silence_threshold_db: float, min_silence_len_ms: int):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        if not isinstance(w, torch.Tensor) or w.ndim != 2 or w.shape[1] == 0:
            return ({"waveform": torch.zeros((1, 1, 0)), "sample_rate": sample_rate},)
        frame_length, hop_length = 2048, 512
        db_threshold = silence_threshold_db
        min_silence_frames = (min_silence_len_ms / 1000) * sample_rate / hop_length
        mono_w = torch.mean(w, dim=0)
        spectrogram = T.Spectrogram(n_fft=frame_length, hop_length=hop_length)(mono_w)
        rms = T.AmplitudeToDB()(spectrogram.pow(2).sum(dim=0).sqrt())
        is_speech = rms > db_threshold
        if not torch.any(is_speech):
            return ({"waveform": torch.zeros((1, w.shape[0], 0)), "sample_rate": sample_rate},)
        speech_segments, in_speech, start_frame = [], False, 0
        for i, frame_is_speech in enumerate(is_speech):
            if frame_is_speech and not in_speech:
                start_frame, in_speech = i, True
            elif not frame_is_speech and in_speech:
                speech_segments.append({'start': start_frame, 'end': i})
                in_speech = False
        if in_speech: speech_segments.append({'start': start_frame, 'end': len(is_speech)})
        if not speech_segments: return ({"waveform": torch.zeros((1, w.shape[0], 0)), "sample_rate": sample_rate},)
        merged_segments = [speech_segments[0]]
        for next_segment in speech_segments[1:]:
            if (next_segment['start'] - merged_segments[-1]['end']) < min_silence_frames:
                merged_segments[-1]['end'] = next_segment['end']
            else:
                merged_segments.append(next_segment)
        final_parts = [w[:, s['start']*hop_length : min(s['end']*hop_length, w.shape[1])] for s in merged_segments]
        processed_w = torch.cat(final_parts, dim=1)
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)


class DeEsser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frequency_hz": ("INT", {"default": 7000, "min": 2000, "max": 12000, "step": 100, "tooltip": "The center frequency of the sibilance to reduce"}),
                "reduction_db": ("FLOAT", {"default": -12.0, "min": -36.0, "max": 0.0, "step": 0.5, "tooltip": "How much to reduce the sibilance by"}),
                "q_factor": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1, "tooltip": "The narrowness of the filter. Higher Q = more surgical cut."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "de_ess"
    CATEGORY = "L3/AudioTools/Processing"

    def de_ess(self, audio: dict, frequency_hz: int, reduction_db: float, q_factor: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        
        # This is the correct approach: use a peaking filter with negative gain to create a notch.
        b0, b1, b2, a0, a1, a2 = _calculate_peaking_coeffs(reduction_db, q_factor, frequency_hz, sample_rate)
        
        processed_w = F.biquad(w, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
        
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class DePlosive:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "cutoff_hz": ("INT", {"default": 80, "min": 40, "max": 300, "step": 5}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "de_plosive"
    CATEGORY = "L3/AudioTools/Processing"
    def de_plosive(self, audio: dict, cutoff_hz: int):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        processed_w = F.highpass_biquad(w, sample_rate, cutoff_hz)
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class ParametricEQ:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "low_cut_hz": ("INT", {"default": 80, "min": 20, "max": 500, "step": 5}), "presence_boost_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5}), "air_boost_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "equalize"
    CATEGORY = "L3/AudioTools/Processing"
    def equalize(self, audio: dict, low_cut_hz: int, presence_boost_db: float, air_boost_db: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        # Low cut (high-pass) is usually available
        w = F.highpass_biquad(w, sample_rate, low_cut_hz)
        # --- THE FIX: Use generic F.biquad for presence (peaking) and air (high-shelf) ---
        b0, b1, b2, a0, a1, a2 = _calculate_peaking_coeffs(presence_boost_db, 0.707, 4000, sample_rate)
        w = F.biquad(w, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
        b0, b1, b2, a0, a1, a2 = _calculate_highshelf_coeffs(air_boost_db, 0.707, 12000, sample_rate)
        w = F.biquad(w, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
        return ({"waveform": w.unsqueeze(0), "sample_rate": sample_rate},)
        
class VocalCompressor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -16.0, "min": -60.0, "max": 0.0, "step": 0.5}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 5.0}),
                "makeup_gain_db": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.5, "tooltip": "Apply gain after compression to make up for volume loss"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "compress"
    CATEGORY = "L3/AudioTools/Processing"
    
    def compress(self, audio: dict, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_gain_db: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        
        # --- THE FIX: Use the robust, manual compressor implementation ---
        processed_w = _manual_compressor(
            w, 
            sample_rate, 
            threshold_db, 
            ratio, 
            attack_ms, 
            release_ms,
            makeup_gain_db
        )
        
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class Reverb:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "room_size": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Size of the simulated room. Controls decay time."}),
                "damping": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "How much high-frequencies are absorbed. Higher = darker sound."}),
                "wet_level": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Volume of the reverb signal."}),
                "dry_level": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Volume of the original signal."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_reverb"
    CATEGORY = "L3/AudioTools/Effects"
    
    def apply_reverb(self, audio: dict, room_size: float, damping: float, wet_level: float, dry_level: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        
        # --- THE FIX: Use the robust, manual reverb implementation ---
        processed_w = _manual_reverb(w, sample_rate, room_size, damping, wet_level, dry_level)
        
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class Delay:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "delay_ms": ("FLOAT", {"default": 500.0, "min": 1.0, "max": 5000.0, "step": 1.0}), "feedback": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.98, "step": 0.01}), "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_delay"
    CATEGORY = "L3/AudioTools/Effects"
    def apply_delay(self, audio: dict, delay_ms: float, feedback: float, mix: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        delay_samples = int(sample_rate * delay_ms / 1000)
        num_channels, num_samples = w.shape
        output = torch.zeros_like(w)
        delay_line = torch.zeros(num_channels, delay_samples, device=w.device)
        for i in range(num_samples):
            delayed_sample = delay_line[:, i % delay_samples]
            input_sample = w[:, i]
            output_sample = input_sample + delayed_sample
            output[:, i] = output_sample
            delay_line[:, i % delay_samples] = input_sample + delayed_sample * feedback
        final_output = (w * (1 - mix) + output * mix).clamp(-1.0, 1.0)
        return ({"waveform": final_output.unsqueeze(0), "sample_rate": sample_rate},)

class PitchTime:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "audio": ("AUDIO",), "pitch_semitones": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}), "tempo_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pitch_time"
    CATEGORY = "L3/AudioTools/Effects"
    def pitch_time(self, audio: dict, pitch_semitones: float, tempo_factor: float):
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        if tempo_factor != 1.0:
            stretcher = T.TimeStretch(n_freq=2048)
            spectrogram = T.Spectrogram(n_fft=4096)(w)
            stretched_spec = stretcher(spectrogram, tempo_factor)
            w = T.GriffinLim(n_fft=4096)(stretched_spec)
        if pitch_semitones != 0:
            w = F.pitch_shift(w, sample_rate, n_steps=pitch_semitones)
        return ({"waveform": w.clamp(-1.0, 1.0).unsqueeze(0), "sample_rate": sample_rate},)
    
# ----------------- UTILITY NODES -----------------

class ConcatenateAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_a": ("AUDIO",),
                "audio_b": ("AUDIO",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat"
    CATEGORY = "L3/AudioTools/Utility"

    def concat(self, audio_a: dict, audio_b: dict):
        w_a, sr_a = audio_a["waveform"][0], audio_a["sample_rate"]
        w_b, sr_b = audio_b["waveform"][0], audio_b["sample_rate"]

        # Resample if necessary
        if sr_a != sr_b:
            resampler = T.Resample(orig_freq=sr_b, new_freq=sr_a)
            w_b = resampler(w_b)
        
        # Concatenate along the time axis (dim=1)
        concatenated_w = torch.cat((w_a, w_b), dim=1)
        return ({"waveform": concatenated_w.unsqueeze(0), "sample_rate": sr_a},)

class StereoPanner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "-1.0 is Left, 0.0 is Center, 1.0 is Right"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pan"
    CATEGORY = "L3/AudioTools/Utility"

    def pan(self, audio: dict, pan: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        
        # If mono, convert to stereo first
        if w.shape[0] == 1:
            w = torch.cat((w, w), dim=0)
            
        # Constant Power Panning Law
        pan_rad = (pan * 0.5 + 0.5) * (math.pi / 2) # Map -1..1 to 0..pi/2
        gain_l = torch.cos(torch.tensor(pan_rad))
        gain_r = torch.sin(torch.tensor(pan_rad))

        panned_w = torch.stack([w[0] * gain_l, w[1] * gain_r])
        return ({"waveform": panned_w.unsqueeze(0), "sample_rate": sr},)

# ----------------- DYNAMICS & REPAIR NODES -----------------

class DeHum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hum_freq": (["60Hz (America)", "50Hz (Europe/Asia)"],),
                "reduction_db": ("FLOAT", {"default": -30.0, "min": -60.0, "max": 0.0, "step": 1.0}),
                "q_factor": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 0.5, "tooltip": "Narrowness of the filter. Higher is more surgical."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "dehum"
    CATEGORY = "L3/AudioTools/Processing"
    
    def dehum(self, audio: dict, hum_freq: str, reduction_db: float, q_factor: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        freq = 60 if "60Hz" in hum_freq else 50
        
        # Apply a notch filter at the fundamental frequency
        b0, b1, b2, a0, a1, a2 = _calculate_peaking_coeffs(reduction_db, q_factor, freq, sr)
        w = F.biquad(w, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
        
        # Apply a second notch at the first harmonic
        b0, b1, b2, a0, a1, a2 = _calculate_peaking_coeffs(reduction_db, q_factor, freq * 2, sr)
        w = F.biquad(w, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
        
        return ({"waveform": w.unsqueeze(0), "sample_rate": sr},)

class NoiseGate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -40.0, "min": -90.0, "max": 0.0, "step": 1.0}),
                "attack_ms": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "release_ms": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 1.0}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "gate"
    CATEGORY = "L3/AudioTools/Processing"

    def gate(self, audio: dict, threshold_db, attack_ms, release_ms):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        
        # Using a simplified version of the manual compressor logic for the gate
        attack_coeff = torch.tensor(math.exp(-1.0 / (sr * (attack_ms / 1000.0))), device=w.device)
        release_coeff = torch.tensor(math.exp(-1.0 / (sr * (release_ms / 1000.0))), device=w.device)
        threshold_lin = 10**(threshold_db / 20.0)
        
        envelope = torch.zeros(w.shape[0], device=w.device)
        gated_w = torch.zeros_like(w)

        for i in range(w.shape[1]):
            input_peak = torch.abs(w[:, i])
            use_attack = input_peak > envelope
            envelope = torch.where(use_attack,
                                   attack_coeff * envelope + (1 - attack_coeff) * input_peak,
                                   release_coeff * envelope + (1 - release_coeff) * input_peak)
            
            is_open = envelope > threshold_lin
            gated_w[:, i] = torch.where(is_open, w[:, i], 0.0)
            
        return ({"waveform": gated_w.unsqueeze(0), "sample_rate": sr},)

# ----------------- ANALYSIS & REACTIVE NODES -----------------

class LoudnessMeter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("loudness_info",)
    FUNCTION = "measure"
    CATEGORY = "L3/AudioTools/Analysis"

    def measure(self, audio: dict):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        
        # pyloudnorm requires numpy array with shape (samples, channels)
        audio_np = w.cpu().numpy().T
        
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_np)
        
        text = f"Integrated Loudness: {loudness:.2f} LUFS"
        print(f"ComfyAudio: {text}")
        return {"ui": {"text": [text]}, "result": (text,)}

class BPMDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            }
        }
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("bpm_info", "beat_events")
    FUNCTION = "detect"
    CATEGORY = "L3/AudioTools/Analysis"

    def detect(self, audio: dict, fps: int):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        # librosa works best with float32 numpy arrays
        audio_mono = torch.mean(w, dim=0).cpu().numpy().astype(np.float32)
        
        tempo, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        duration_sec = len(audio_mono) / sr
        total_video_frames = int(duration_sec * fps)
        
        beat_events = np.zeros(total_video_frames, dtype=np.float32)
        for t in beat_times:
            frame_idx = int(t * fps)
            if frame_idx < total_video_frames:
                beat_events[frame_idx] = 1.0
        
        # --- THE FIX ---
        # The 'tempo' variable is a numpy array (e.g., array([120.0])). We need to get the first element.
        bpm_info = f"Estimated BPM: {tempo[0]:.2f}"
        
        # ComfyUI handles list-to-batch conversion for primitive types
        return (bpm_info, beat_events.tolist())

class AudioReactiveParameter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Smooths the output. 0=no smoothing, 1=max smoothing."}),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("envelope",)
    FUNCTION = "analyze"
    CATEGORY = "L3/AudioTools/Analysis"

    def analyze(self, audio: dict, fps: int, smoothing: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        audio_mono = torch.mean(w, dim=0).cpu().numpy()
        
        # Get RMS energy envelope from librosa
        rms = librosa.feature.rms(y=audio_mono)[0]
        
        # Resample the RMS envelope to match the number of video frames
        duration_sec = len(audio_mono) / sr
        total_video_frames = int(duration_sec * fps)
        
        # Create time axes for original RMS and target video frames
        rms_times = librosa.times_like(rms, sr=sr)
        video_times = np.linspace(0, duration_sec, total_video_frames)
        
        # Interpolate to match video frame rate
        envelope = np.interp(video_times, rms_times, rms)
        
        # Normalize to 0-1 range
        if np.max(envelope) > 0:
            envelope /= np.max(envelope)
        
        # Apply smoothing (exponential moving average)
        if smoothing > 0:
            alpha = 1.0 - smoothing
            for i in range(1, len(envelope)):
                envelope[i] = alpha * envelope[i] + (1 - alpha) * envelope[i-1]

        return (envelope.tolist(),)
    
class FadeIn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "duration_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade"
    CATEGORY = "L3/AudioTools/Effects"

    def fade(self, audio: dict, duration_seconds: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        num_channels, total_samples = w.shape
        
        fade_samples = min(int(duration_seconds * sr), total_samples)
        
        if fade_samples > 0:
            fade_curve = torch.linspace(0.0, 1.0, fade_samples, device=w.device).unsqueeze(0)
            w[:, :fade_samples] *= fade_curve
            
        return ({"waveform": w.unsqueeze(0), "sample_rate": sr},)

class FadeOut:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "duration_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade"
    CATEGORY = "L3/AudioTools/Effects"

    def fade(self, audio: dict, duration_seconds: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        num_channels, total_samples = w.shape
        
        fade_samples = min(int(duration_seconds * sr), total_samples)
        
        if fade_samples > 0:
            fade_curve = torch.linspace(1.0, 0.0, fade_samples, device=w.device).unsqueeze(0)
            w[:, -fade_samples:] *= fade_curve

        return ({"waveform": w.unsqueeze(0), "sample_rate": sr},)

class PadAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pad_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "pad_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pad"
    CATEGORY = "L3/AudioTools/Utility"

    def pad(self, audio: dict, pad_start_seconds: float, pad_end_seconds: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        num_channels, total_samples = w.shape

        if pad_start_seconds > 0:
            pad_samples = int(pad_start_seconds * sr)
            start_padding = torch.zeros((num_channels, pad_samples), device=w.device)
            w = torch.cat((start_padding, w), dim=1)

        if pad_end_seconds > 0:
            pad_samples = int(pad_end_seconds * sr)
            end_padding = torch.zeros((num_channels, pad_samples), device=w.device)
            w = torch.cat((w, end_padding), dim=1)
            
        return ({"waveform": w.unsqueeze(0), "sample_rate": sr},)