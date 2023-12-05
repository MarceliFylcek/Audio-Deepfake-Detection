import torchaudio
import sounddevice as sd
import torch

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

rir_f = 'augmentation samples/room_impulse_responses/rir__1_98_2_8.flac'
noise_f = 'augmentation samples/background_noises/noise.flac'

timestamps = get_rir_timestamps(rir_f)

rir_audio, rir_sample_rate = torchaudio.load(rir_f)
noise_audio, noise_sample_rate = torchaudio.load(noise_f)
snr = torch.tensor([10])

def room_reverb(speech_audio,audio_sample_rate, rir_raw=rir_audio, rir_sample_rate=rir_sample_rate, rir_timestamps=timestamps):
    
    start, stop = rir_timestamps
    rir = rir_raw[:, int(rir_sample_rate * start) : int(rir_sample_rate * stop)] 
    rir = rir / torch.linalg.vector_norm(rir, ord=2)

    rir = torchaudio.transforms.Resample(rir_sample_rate, audio_sample_rate)(rir)

    if rir.shape[0] > speech_audio.shape[0]:
        rir = rir[:speech_audio.shape[0], :]

    if rir.shape[1] > speech_audio.shape[1]:
        # Truncate the rir tensor if it's longer than the speech tensor
        rir = rir[:, :speech_audio.shape[1]]
    elif rir.shape[1] < speech_audio.shape[1]:
        # Zero-pad the rir tensor if it's shorter than the speech tensor
        padding = speech_audio.shape[1] - rir.shape[1]
        rir = torch.nn.functional.pad(rir, (0, padding))

    augmented = torchaudio.functional.fftconvolve(speech_audio, rir)

    if augmented.shape[1] > speech_audio.shape[1]:
        augmented = augmented[:, :speech_audio.shape[1]]

    return augmented


def add_noise(speech_audio,audio_sample_rate, noise_raw=noise_audio, noise_sample_rate = noise_sample_rate, snr=snr):

    noise = torchaudio.transforms.Resample(noise_sample_rate, audio_sample_rate)(noise_raw)
    noise = noise[:, : speech_audio.shape[1]]

    if noise.shape[0] > speech_audio.shape[0]:
        noise = noise[:speech_audio.shape[0], :]

    # Zero-pad or truncate the noise tensor to match the size of the speech tensor
    if noise.shape[1] > speech_audio.shape[1]:
        # Truncate the noise tensor if it's longer than the speech tensor
        noise = noise[:, :speech_audio.shape[1]]
    elif noise.shape[1] < speech_audio.shape[1]:
        # Zero-pad the noise tensor if it's shorter than the speech tensor
        padding = speech_audio.shape[1] - noise.shape[1]
        noise = torch.nn.functional.pad(noise, (0, padding))

    noisy_speech = torchaudio.functional.add_noise(speech_audio, noise, snr)

    return noisy_speech


if __name__ == '__main__':
    from utils import play, change_audio_len, plot_waveform # helper functions for developement
    rir_f = 'augmentation samples/room_impulse_responses/rir__1_98_2_8.flac'
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