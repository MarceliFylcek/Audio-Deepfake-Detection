import torchaudio
import sounddevice as sd
import torch
import random

def get_rir_timestamps(rir_filename):
    """Gets the timestamps of room inpulse response in audio file

    Args:
        rir_filename (str): path to rir audio file

    Returns:
        tuple: timestamps
    """
    split = rir_filename.split('.')[-2].split('__')[-1].split('_')
    start, end = float('.'.join(split[0:2])), float('.'.join(split[2:]))
    return (start, end)


def room_reverb(speech, rir_raw, audio_sample_rate, rir_sample_rate, rir_timestamps):
    """Applies room reverbation to audio based on provided room impulse response

    Args:
        speech (tensor): speech audio
        rir_raw (tensor): room impulse response audio
        audio_sample_rate (int): speech audio sample rate
        rir_sample_rate (int): room impulse response sample rate
        rir_timestamps (tuple): room impulse response timestamps in provided audio

    Returns:
        tensor: augmented speech audio
    """

    # get impulse response from room recording and normalize it
    start, stop = rir_timestamps
    rir = rir_raw[:, int(rir_sample_rate * start) : int(rir_sample_rate * stop)] 
    rir = rir / torch.linalg.vector_norm(rir, ord=2)

    rir = torchaudio.transforms.Resample(rir_sample_rate, audio_sample_rate)(rir)
    augmented = torchaudio.functional.fftconvolve(speech, rir)

    return augmented


def add_noise(speech, noise_raw, audio_sample_rate, noise_sample_rate, snr):
    """Adds noise based on provided noise audio

    Args:
        speech (tensor): speech audio
        noise_raw (tensor): noise audio
        audio_sample_rate (int): speech audio sample rate
        noise_sample_rate (int): noise audio sample rate
        snr (tensor): signal to noise ratio

    Returns:
        tensor: speech audio with added noise
    """

    noise = torchaudio.transforms.Resample(noise_sample_rate, audio_sample_rate)(noise_raw)
    noise = noise[:, : speech.shape[1]]

    noisy_speech = torchaudio.functional.add_noise(speech, noise, snr)

    return noisy_speech

def Speech_audio(audio_f):
    speech_audio, audio_sample_rate = torchaudio.load(audio_f)
    speech_audio = change_audio_len(speech_audio, audio_sample_rate, 4000)
    return speech_audio, audio_sample_rate

def reverb_audio(audio_f, rir_f='augmentation samples/room_impulse_responses/rir__1_98_2_8.flac'):
    rir_audio, rir_sample_rate = torchaudio.load(rir_f)
    timestamps = get_rir_timestamps(rir_f)
    reverbed = room_reverb(audio_f, rir_audio, 16000, rir_sample_rate, timestamps)
    return reverbed, audio_sample_rate

def noisy_audio(audio_f, noise_f='augmentation samples/background_noises/noise.flac'):
    noise_audio, noise_sample_rate = torchaudio.load(noise_f)
    snr = torch.tensor([10])
    noisy = add_noise(audio_f, noise_audio, 16000, noise_sample_rate, snr)
    return noisy, audio_sample_rate

if __name__ == '__main__':
    from utils import play, change_audio_len, plot_waveform # helper functions for developement
    rir_f = 'augmentation samples/room_impulse_responses/rir__1_98_2_8.flac'
    audio_f = 'resources/train/real/198-0004.flac'
    noise_f = 'augmentation samples/background_noises/noise.flac'
    
    timestamps = get_rir_timestamps(rir_f)

    speech_audio, audio_sample_rate = torchaudio.load(audio_f)
    rir_audio, rir_sample_rate = torchaudio.load(rir_f)
    noise_audio, noise_sample_rate = torchaudio.load(noise_f)
    snr = torch.tensor([10])

    speech_audio = change_audio_len(speech_audio, audio_sample_rate, 4000)
    reverbed = room_reverb(speech_audio, rir_audio, audio_sample_rate, rir_sample_rate, timestamps)
    noisy = add_noise(speech_audio, noise_audio, audio_sample_rate, noise_sample_rate, snr)

    play(speech_audio, audio_sample_rate)
    play(noisy, audio_sample_rate)
    play(reverbed, audio_sample_rate)
