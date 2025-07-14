import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import math
import numpy as np
import librosa
import pyloudnorm as pyln


def _calculate_peaking_coeffs(gain_db, q, center_freq, sample_rate):
    A = 10**(gain_db / 40.0)
    w0 = 2.0 * math.pi * center_freq / sample_rate
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0, b1, b2 = 1.0 + alpha * A, -2.0 * cos_w0, 1.0 - alpha * A
    a0, a1, a2 = 1.0 + alpha / A, -2.0 * cos_w0, 1.0 - alpha / A
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
    attack_coeff = torch.tensor(math.exp(-1.0 / (sample_rate * (attack_ms / 1000.0))), device=waveform.device)
    release_coeff = torch.tensor(math.exp(-1.0 / (sample_rate * (release_ms / 1000.0))), device=waveform.device)
    makeup_gain_lin = 10**(makeup_gain_db / 20.0)
    num_channels, num_samples = waveform.shape
    envelope = torch.zeros(num_channels, device=waveform.device)
    gain = torch.ones(num_channels, device=waveform.device)
    output_waveform = torch.zeros_like(waveform)
    for i in range(num_samples):
        input_peak = torch.max(torch.abs(waveform[:, i]))
        use_attack = input_peak > envelope
        envelope = torch.where(use_attack, attack_coeff * envelope + (1 - attack_coeff) * input_peak, release_coeff * envelope + (1 - release_coeff) * input_peak)
        envelope_db = 20 * torch.log10(envelope + 1e-9)
        above_threshold = envelope_db > threshold_db
        gain_reduction_db = (threshold_db - envelope_db) * (1.0 - (1.0 / ratio))
        reduced_gain = 10**(gain_reduction_db / 20.0)
        gain = torch.where(above_threshold, reduced_gain, 1.0)
        output_waveform[:, i] = waveform[:, i] * gain * makeup_gain_lin
    return output_waveform.clamp(-1.0, 1.0)

def _manual_reverb(waveform, sample_rate, room_size, damping, wet_level, dry_level):
    num_channels, num_samples = waveform.shape
    device = waveform.device
    room_size_factor = room_size / 100.0
    decay_time = 0.1 + room_size_factor * 0.9
    feedback_gain = 0.5 + room_size_factor * 0.49
    damping_factor = damping / 100.0
    damping_coeff = 1.0 - damping_factor * 0.7
    comb_delays = [int(d * sample_rate) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
    allpass_delays = [int(d * sample_rate) for d in [0.005, 0.0017]]
    allpass_gain = 0.7
    comb_buffers = [torch.zeros(num_channels, d, device=device) for d in comb_delays]
    allpass_buffers = [torch.zeros(num_channels, d, device=device) for d in allpass_delays]
    comb_pos = [0] * 4
    allpass_pos = [0] * 2
    output_waveform = torch.zeros_like(waveform)
    for i in range(num_samples):
        s_in = waveform[:, i]
        s_comb_out = torch.zeros(num_channels, device=device)
        for j in range(len(comb_delays)):
            delayed_sample = comb_buffers[j][:, comb_pos[j]]
            damped_sample = delayed_sample * damping_coeff
            comb_buffers[j][:, comb_pos[j]] = s_in + damped_sample * feedback_gain
            comb_pos[j] = (comb_pos[j] + 1) % comb_delays[j]
            s_comb_out += delayed_sample
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

# ----------------- BATCH-COMPATIBLE PROCESSING NODES -----------------

class AmplifyGain:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to apply gain to."}), 
                        "gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, "tooltip": "Amount of gain in decibels (dB) to apply. Positive values amplify, negative values attenuate."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_gain"
    CATEGORY = "AudioTools/Processing"
    def apply_gain(self, audio: dict, gain_db: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            processed_w = F.gain(w, gain_db).clamp(-1.0, 1.0)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class NormalizeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to normalize."}), 
                        "target_level_db": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1, "tooltip": "The target peak volume in decibels (dB). A value of 0.0 is maximum, but -1.0 is a common target to avoid clipping."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize"
    CATEGORY = "AudioTools/Processing"
    def normalize(self, audio: dict, target_level_db: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            current_peak = 20 * torch.log10(torch.max(torch.abs(w)))
            gain_to_apply = 0 if torch.isinf(current_peak) else target_level_db - current_peak
            processed_w = F.gain(w, gain_to_apply).clamp(-1.0, 1.0)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

# NOTE: Mix and Concatenate are intentionally left as single-item processors
# to avoid ambiguity. They will operate on the first item of each batch.
class MixAudio:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
                        "audio_1": ("AUDIO", {"tooltip": "The first audio track."}), 
                        "audio_2": ("AUDIO", {"tooltip": "The second audio track."}), 
                        "gain_1_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 6.0, "step": 0.1, "tooltip": "Gain in dB for the first audio track."}), 
                        "gain_2_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 6.0, "step": 0.1, "tooltip": "Gain in dB for the second audio track."}), } }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "mix"
    CATEGORY = "AudioTools/Processing"
    def mix(self, audio_1: dict, audio_2: dict, gain_1_db: float, gain_2_db: float):
        w1, sr1 = audio_1["waveform"][0], audio_1["sample_rate"]
        w2, sr2 = audio_2["waveform"][0], audio_2["sample_rate"]
        if sr1 != sr2: w2 = T.Resample(orig_freq=sr2, new_freq=sr1)(w2)
        w1, w2 = F.gain(w1, gain_1_db), F.gain(w2, gain_2_db)
        len1, len2 = w1.shape[1], w2.shape[1]
        if len1 > len2: w2 = torch.nn.functional.pad(w2, (0, len1 - len2))
        elif len2 > len1: w1 = torch.nn.functional.pad(w1, (0, len2 - len1))
        mixed = (w1 + w2).clamp(-1.0, 1.0)
        return ({"waveform": mixed.unsqueeze(0), "sample_rate": sr1},)

class RemoveSilence:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
                        "audio": ("AUDIO", {"tooltip": "The audio clip from which to remove silent sections."}), 
                        "silence_threshold_db": ("FLOAT", {"default": -40.0, "min": -90.0, "max": 0.0, "step": 1.0, "tooltip": "The volume level (in dB) below which audio is considered silent."}), 
                        "min_silence_len_ms": ("INT", {"default": 500, "min": 50, "max": 5000, "step": 50, "tooltip": "The minimum duration (in milliseconds) of silence to be removed."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "remove_silence"
    CATEGORY = "AudioTools/Processing"
    def remove_silence(self, audio: dict, silence_threshold_db: float, min_silence_len_ms: int):
        # NOTE: This is a complex operation. Batch processing this node could lead to
        # highly variable lengths, making re-batching difficult. It is left as a
        # single-item processor for predictability. Use a loop with GetAudioFromList.
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        if not isinstance(w, torch.Tensor) or w.ndim != 2 or w.shape[1] == 0: return ({"waveform": torch.zeros((1, 1, 0)), "sample_rate": sample_rate},)
        frame_length, hop_length = 2048, 512
        min_silence_frames = (min_silence_len_ms / 1000) * sample_rate / hop_length
        mono_w = torch.mean(w, dim=0)
        spectrogram = T.Spectrogram(n_fft=frame_length, hop_length=hop_length)(mono_w)
        rms = T.AmplitudeToDB()(spectrogram.pow(2).sum(dim=0).sqrt())
        is_speech = rms > silence_threshold_db
        if not torch.any(is_speech): return ({"waveform": torch.zeros((1, w.shape[0], 0)), "sample_rate": sample_rate},)
        speech_segments, in_speech, start_frame = [], False, 0
        for i, frame_is_speech in enumerate(is_speech):
            if frame_is_speech and not in_speech: start_frame, in_speech = i, True
            elif not frame_is_speech and in_speech: speech_segments.append({'start': start_frame, 'end': i}); in_speech = False
        if in_speech: speech_segments.append({'start': start_frame, 'end': len(is_speech)})
        if not speech_segments: return ({"waveform": torch.zeros((1, w.shape[0], 0)), "sample_rate": sample_rate},)
        merged_segments = [speech_segments[0]]
        for next_segment in speech_segments[1:]:
            if (next_segment['start'] - merged_segments[-1]['end']) < min_silence_frames: merged_segments[-1]['end'] = next_segment['end']
            else: merged_segments.append(next_segment)
        final_parts = [w[:, s['start']*hop_length : min(s['end']*hop_length, w.shape[1])] for s in merged_segments]
        processed_w = torch.cat(final_parts, dim=1)
        return ({"waveform": processed_w.unsqueeze(0), "sample_rate": sample_rate},)

class DeEsser:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to de-ess."}), 
                        "frequency_hz": ("INT", {"default": 7000, "min": 2000, "max": 12000, "step": 100, "tooltip": "The center frequency of sibilance to target (typically 5-8 kHz)."}), 
                        "reduction_db": ("FLOAT", {"default": -12.0, "min": -36.0, "max": 0.0, "step": 0.5, "tooltip": "The amount of gain reduction (in dB) to apply at the target frequency."}), 
                        "q_factor": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1, "tooltip": "The width of the frequency band to affect. Higher Q is narrower."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "de_ess"
    CATEGORY = "AudioTools/Processing"
    def de_ess(self, audio: dict, frequency_hz: int, reduction_db: float, q_factor: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        b0, b1, b2, a0, a1, a2 = _calculate_peaking_coeffs(reduction_db, q_factor, frequency_hz, sample_rate)
        # --- BATCH PROCESSING ---
        for w in w_batch:
            processed_w = F.biquad(w, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class DePlosive:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to de-plosive (low-cut)."}), 
                        "cutoff_hz": ("INT", {"default": 80, "min": 40, "max": 300, "step": 5, "tooltip": "The cutoff frequency for the high-pass filter. Frequencies below this will be rolled off."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "de_plosive"
    CATEGORY = "AudioTools/Processing"
    def de_plosive(self, audio: dict, cutoff_hz: int):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            processed_w = F.highpass_biquad(w, sample_rate, cutoff_hz)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class ParametricEQ:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to equalize."}), 
                        "low_cut_hz": ("INT", {"default": 80, "min": 20, "max": 500, "step": 5, "tooltip": "Low-cut (high-pass) filter to remove rumble. 80-120Hz is common for voice."}), 
                        "presence_boost_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "tooltip": "Boost/cut for vocal presence (around 4kHz)."}), 
                        "air_boost_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "tooltip": "High-shelf boost/cut for 'air' and clarity (around 12kHz)."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "equalize"
    CATEGORY = "AudioTools/Processing"
    def equalize(self, audio: dict, low_cut_hz: int, presence_boost_db: float, air_boost_db: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_proc = F.highpass_biquad(w, sample_rate, low_cut_hz)
            b0, b1, b2, a0, a1, a2 = _calculate_peaking_coeffs(presence_boost_db, 0.707, 4000, sample_rate)
            w_proc = F.biquad(w_proc, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
            b0, b1, b2, a0, a1, a2 = _calculate_highshelf_coeffs(air_boost_db, 0.707, 12000, sample_rate)
            w_proc = F.biquad(w_proc, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2)
            processed_list.append(w_proc)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class VocalCompressor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to compress."}), 
                        "threshold_db": ("FLOAT", {"default": -16.0, "min": -60.0, "max": 0.0, "step": 0.5, "tooltip": "The volume level (dB) at which the compressor starts working."}), 
                        "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "The amount of gain reduction (e.g., 4.0 means a 4:1 ratio)."}), 
                        "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "tooltip": "How quickly (in ms) the compressor reacts to loud sounds."}), 
                        "release_ms": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 5.0, "tooltip": "How quickly (in ms) the compressor stops after the sound falls below the threshold."}), 
                        "makeup_gain_db": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.5, "tooltip": "Volume boost to apply after compression to make up for the reduced level."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "compress"
    CATEGORY = "AudioTools/Processing"
    def compress(self, audio: dict, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_gain_db: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            processed_w = _manual_compressor(w, sample_rate, threshold_db, ratio, attack_ms, release_ms, makeup_gain_db)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class Reverb:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to apply reverb to."}), 
                        "room_size": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0, 
                                                "tooltip": "The perceived size of the reverberant space (0-100)."}), 
                                                "damping": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "How much the high frequencies are absorbed in the reverb tails (0-100)."}), 
                                                "wet_level": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The volume of the reverberated (wet) signal."}), 
                                                "dry_level": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The volume of the original (dry) signal."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_reverb"
    CATEGORY = "AudioTools/Effects"
    def apply_reverb(self, audio: dict, room_size: float, damping: float, wet_level: float, dry_level: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            processed_w = _manual_reverb(w, sample_rate, room_size, damping, wet_level, dry_level)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class Delay:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to apply delay to."}), 
                        "delay_ms": ("FLOAT", {"default": 500.0, "min": 1.0, "max": 5000.0, "step": 1.0, "tooltip": "The time (in milliseconds) between each echo."}), 
                        "feedback": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.98, "step": 0.01, "tooltip": "How much of the delayed signal is fed back into the delay line, creating more echoes."}), 
                        "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The balance between the original (dry) and delayed (wet) signal. 0.0 is all dry, 1.0 is all wet."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_delay"
    CATEGORY = "AudioTools/Effects"
    def apply_delay(self, audio: dict, delay_ms: float, feedback: float, mix: float):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
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
            processed_w = (w * (1 - mix) + output * mix).clamp(-1.0, 1.0)
            processed_list.append(processed_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)

class PitchTime:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to pitch shift or time stretch."}), 
                        "pitch_semitones": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, "tooltip": "The number of semitones to shift the pitch up or down."}), 
                        "tempo_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "The factor by which to change the tempo. >1.0 is faster, <1.0 is slower."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pitch_time"
    CATEGORY = "AudioTools/Effects"
    def pitch_time(self, audio: dict, pitch_semitones: float, tempo_factor: float):
        # NOTE: Pitch/Time is very slow and would be even slower on a batch.
        # It's also likely to create variable lengths. Left as single-item processor.
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        w_proc = w.clone()
        if tempo_factor != 1.0:
            stretcher = T.TimeStretch(n_freq=2048)
            spectrogram = T.Spectrogram(n_fft=4096)(w_proc)
            stretched_spec = stretcher(spectrogram, tempo_factor)
            w_proc = T.GriffinLim(n_fft=4096)(stretched_spec)
        if pitch_semitones != 0:
            w_proc = F.pitch_shift(w_proc, sample_rate, n_steps=pitch_semitones)
        return ({"waveform": w_proc.clamp(-1.0, 1.0).unsqueeze(0), "sample_rate": sample_rate},)

class TrimAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio clip to trim."}), 
                        "trim_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Number of seconds to cut from the beginning of the audio."}), 
                        "trim_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Number of seconds to cut from the end of the audio."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "trim"
    CATEGORY = "AudioTools/Processing"
    def trim(self, audio: dict, trim_start_seconds: float, trim_end_seconds: float):
        # NOTE: Trimming a batch would result in uneven lengths.
        # This node is designed to operate on a single audio clip.
        w, sample_rate = audio["waveform"][0], audio["sample_rate"]
        num_channels, total_samples = w.shape
        start_sample = max(0, int(trim_start_seconds * sample_rate))
        end_sample = total_samples - int(trim_end_seconds * sample_rate)
        end_sample = min(total_samples, max(start_sample, end_sample))
        if start_sample >= end_sample:
            return ({"waveform": torch.zeros((1, num_channels, 0), device=w.device), "sample_rate": sample_rate},)
        else:
            return ({"waveform": w[:, start_sample:end_sample].unsqueeze(0), "sample_rate": sample_rate},)

# ----------------- UTILITY NODES (BATCH-COMPATIBLE) -----------------

class ConcatenateAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio_a": ("AUDIO", {"tooltip": "The first audio clip (will be the beginning of the result)."}), 
                        "audio_b": ("AUDIO", {"tooltip": "The second audio clip (will be appended to the end of the first)."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat"
    CATEGORY = "AudioTools/Utility"
    def concat(self, audio_a: dict, audio_b: dict):
        # NOTE: Concatenates the first item from each batch end-to-end.
        w_a, sr_a = audio_a["waveform"][0], audio_a["sample_rate"]
        w_b, sr_b = audio_b["waveform"][0], audio_b["sample_rate"]
        if sr_a != sr_b: w_b = T.Resample(orig_freq=sr_b, new_freq=sr_a)(w_b)
        concatenated_w = torch.cat((w_a, w_b), dim=1)
        return ({"waveform": concatenated_w.unsqueeze(0), "sample_rate": sr_a},)

class StereoPanner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to pan."}), 
                        "pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Stereo position. -1.0 is hard left, 1.0 is hard right, 0.0 is center."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pan"
    CATEGORY = "AudioTools/Utility"
    def pan(self, audio: dict, pan: float):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        pan_rad = (pan * 0.5 + 0.5) * (math.pi / 2)
        gain_l = torch.cos(torch.tensor(pan_rad))
        gain_r = torch.sin(torch.tensor(pan_rad))
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_proc = w.clone()
            if w_proc.shape[0] == 1: w_proc = torch.cat((w_proc, w_proc), dim=0)
            panned_w = torch.stack([w_proc[0] * gain_l, w_proc[1] * gain_r])
            processed_list.append(panned_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)

class DeHum:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to de-hum."}), 
                        "hum_freq": (["60Hz (America)", "50Hz (Europe/Asia)"], {"tooltip": "The fundamental frequency of the electrical hum to remove."}), 
                        "reduction_db": ("FLOAT", {"default": -30.0, "min": -60.0, "max": 0.0, "step": 1.0, "tooltip": "The amount of gain reduction (dB) to apply to the hum frequency and its first harmonic."}), 
                        "q_factor": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 0.5, "tooltip": "The narrowness of the filter. A high Q value is needed to target only the hum."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "dehum"
    CATEGORY = "AudioTools/Processing"
    def dehum(self, audio: dict, hum_freq: str, reduction_db: float, q_factor: float):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        freq = 60 if "60Hz" in hum_freq else 50
        b0_1, b1_1, b2_1, a0_1, a1_1, a2_1 = _calculate_peaking_coeffs(reduction_db, q_factor, freq, sr)
        b0_2, b1_2, b2_2, a0_2, a1_2, a2_2 = _calculate_peaking_coeffs(reduction_db, q_factor, freq * 2, sr)
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_proc = F.biquad(w, b0=b0_1, b1=b1_1, b2=b2_1, a0=a0_1, a1=a1_1, a2=a2_1)
            w_proc = F.biquad(w_proc, b0=b0_2, b1=b1_2, b2=b2_2, a0=a0_2, a1=a1_2, a2=a2_2)
            processed_list.append(w_proc)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)

class NoiseGate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to apply the noise gate to."}), 
                        "threshold_db": ("FLOAT", {"default": -40.0, "min": -90.0, "max": 0.0, "step": 1.0, "tooltip": "The volume level (dB) below which the gate will close and silence the audio."}), 
                        "attack_ms": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0, "tooltip": "How quickly (in ms) the gate opens when the signal exceeds the threshold."}), 
                        "release_ms": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 1.0, "tooltip": "How quickly (in ms) the gate closes after the signal falls below the threshold."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "gate"
    CATEGORY = "AudioTools/Processing"
    def gate(self, audio: dict, threshold_db, attack_ms, release_ms):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            attack_coeff = torch.tensor(math.exp(-1.0 / (sr * (attack_ms / 1000.0))), device=w.device)
            release_coeff = torch.tensor(math.exp(-1.0 / (sr * (release_ms / 1000.0))), device=w.device)
            threshold_lin = 10**(threshold_db / 20.0)
            envelope = torch.zeros(w.shape[0], device=w.device)
            gated_w = torch.zeros_like(w)
            for i in range(w.shape[1]):
                input_peak = torch.abs(w[:, i])
                use_attack = input_peak > envelope
                envelope = torch.where(use_attack, attack_coeff * envelope + (1 - attack_coeff) * input_peak, release_coeff * envelope + (1 - release_coeff) * input_peak)
                is_open = envelope > threshold_lin
                gated_w[:, i] = torch.where(is_open, w[:, i], 0.0)
            processed_list.append(gated_w)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)

# ----------------- ANALYSIS & REACTIVE NODES (BATCH-INCOMPATIBLE) -----------------
# These nodes analyze a single audio clip to produce a single result.

class LoudnessMeter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio clip to measure for loudness."}),}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("loudness_info",)
    FUNCTION = "measure"
    CATEGORY = "AudioTools/Analysis"
    def measure(self, audio: dict):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        audio_np = w.cpu().numpy().T
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_np)
        text = f"Integrated Loudness: {loudness:.2f} LUFS"
        return {"ui": {"text": [text]}, "result": (text,)}

class BPMDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio clip to detect the tempo from."}), 
                        "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "The target frames per second to sync the beat events list with."}),}}
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("bpm_info", "beat_events")
    FUNCTION = "detect"
    CATEGORY = "AudioTools/Analysis"
    def detect(self, audio: dict, fps: int):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        audio_mono = torch.mean(w, dim=0).cpu().numpy().astype(np.float32)
        tempo, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        duration_sec = len(audio_mono) / sr
        total_video_frames = int(duration_sec * fps)
        beat_events = np.zeros(total_video_frames, dtype=np.float32)
        for t in beat_times:
            frame_idx = int(t * fps)
            if frame_idx < total_video_frames: beat_events[frame_idx] = 1.0
        bpm_info = f"Estimated BPM: {tempo[0]:.2f}"
        return (bpm_info, beat_events.tolist())

class AudioReactiveParameter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO", {"tooltip": "The audio clip to analyze for its volume envelope."}), 
                    "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "The target frames per second to sync the envelope list with."}), 
                    "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amount of smoothing to apply to the envelope. 0 is no smoothing, 1 is maximum smoothing."}),}}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("envelope",)
    FUNCTION = "analyze"
    CATEGORY = "AudioTools/Analysis"
    def analyze(self, audio: dict, fps: int, smoothing: float):
        w, sr = audio["waveform"][0], audio["sample_rate"]
        audio_mono = torch.mean(w, dim=0).cpu().numpy()
        rms = librosa.feature.rms(y=audio_mono)[0]
        duration_sec = len(audio_mono) / sr
        total_video_frames = int(duration_sec * fps)
        rms_times = librosa.times_like(rms, sr=sr)
        video_times = np.linspace(0, duration_sec, total_video_frames)
        envelope = np.interp(video_times, rms_times, rms)
        if np.max(envelope) > 0: envelope /= np.max(envelope)
        if smoothing > 0:
            alpha = 1.0 - smoothing
            for i in range(1, len(envelope)): envelope[i] = alpha * envelope[i] + (1 - alpha) * envelope[i-1]
        return (envelope.tolist(),)

class FadeIn:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to apply a fade-in to."}), 
                        "duration_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "The duration of the fade-in effect in seconds."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade"
    CATEGORY = "AudioTools/Effects"
    def fade(self, audio: dict, duration_seconds: float):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_copy = w.clone()
            num_channels, total_samples = w_copy.shape
            fade_samples = min(int(duration_seconds * sr), total_samples)
            if fade_samples > 0:
                fade_curve = torch.linspace(0.0, 1.0, fade_samples, device=w_copy.device).unsqueeze(0)
                w_copy[:, :fade_samples] *= fade_curve
            processed_list.append(w_copy)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)

class FadeOut:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio to apply a fade-out to."}), 
                        "duration_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "The duration of the fade-out effect in seconds."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade"
    CATEGORY = "AudioTools/Effects"
    def fade(self, audio: dict, duration_seconds: float):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_copy = w.clone()
            num_channels, total_samples = w_copy.shape
            fade_samples = min(int(duration_seconds * sr), total_samples)
            if fade_samples > 0:
                fade_curve = torch.linspace(1.0, 0.0, fade_samples, device=w_copy.device).unsqueeze(0)
                w_copy[:, -fade_samples:] *= fade_curve
            processed_list.append(w_copy)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)

class PadAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "audio": ("AUDIO", {"tooltip": "The audio clip to pad with silence."}), 
                        "pad_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Duration of silence to add to the beginning of the audio."}), 
                        "pad_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Duration of silence to add to the end of the audio."}),}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pad"
    CATEGORY = "AudioTools/Utility"
    def pad(self, audio: dict, pad_start_seconds: float, pad_end_seconds: float):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_proc = w.clone()
            num_channels, total_samples = w_proc.shape
            if pad_start_seconds > 0:
                F = int(pad_start_seconds * sr)
                start_padding = torch.zeros((num_channels, pad_samples), device=w_proc.device)
                w_proc = torch.cat((start_padding, w_proc), dim=1)
            if pad_end_seconds > 0:
                pad_samples = int(pad_end_seconds * sr)
                end_padding = torch.zeros((num_channels, pad_samples), device=w_proc.device)
                w_proc = torch.cat((w_proc, end_padding), dim=1)
            processed_list.append(w_proc)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)
    
class StandardizeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio to standardize."}),
                "channel_layout": (["mono", "stereo"], {"default": "mono", "tooltip": "Convert audio to mono (single channel) or ensure it is stereo (two channels)."}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "standardize"
    CATEGORY = "AudioTools/Processing"

    def standardize(self, audio: dict, channel_layout: str):
        w_batch, sample_rate = audio["waveform"], audio["sample_rate"]
        
        if w_batch.nelement() == 0:
            return ({"waveform": torch.zeros((0, 1, 1)), "sample_rate": sample_rate},)

        processed_list = []
        # --- BATCH PROCESSING ---
        for w in w_batch:
            w_proc = w.clone()
            
            # 1. Standardize Data Type to float32
            if w_proc.dtype != torch.float32:
                print(f"ComfyAudio (Standardize): Converting waveform from {w_proc.dtype} to torch.float32.")
                w_proc = w_proc.to(torch.float32)

            # 2. Standardize Channel Layout
            if channel_layout == "mono":
                if w_proc.shape[0] > 1:
                    print(f"ComfyAudio (Standardize): Converting audio to mono.")
                    w_proc = torch.mean(w_proc, dim=0, keepdim=True)
            elif channel_layout == "stereo":
                if w_proc.shape[0] == 1:
                    print(f"ComfyAudio (Standardize): Converting mono audio to stereo.")
                    w_proc = torch.cat((w_proc, w_proc), dim=0)
                elif w_proc.shape[0] > 2:
                    print(f"ComfyAudio (Standardize): Warning - audio has more than 2 channels. Taking first 2 for stereo.")
                    w_proc = w_proc[:2, :]

            processed_list.append(w_proc)
            
        return ({"waveform": torch.stack(processed_list), "sample_rate": sample_rate},)