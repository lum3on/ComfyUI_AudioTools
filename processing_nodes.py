# File: processing_nodes.py (Final Compatible Version)

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import math

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
    CATEGORY = "L3/AudioTools/Processing"
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