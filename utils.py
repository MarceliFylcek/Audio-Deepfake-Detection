import os
from dataset import FakeAudioDataset
from torch.utils.data import DataLoader
import traceback
import sounddevice as sd
import torch


def get_dataloader(dataset_path, batch_size, melspect_params, transform, shuffle=True):
    real_folder = os.path.join(dataset_path, "real")
    fake_folder = os.path.join(dataset_path, "fake")

    #! Cuts only first 4 seconds of the recording as of now
    dataset = FakeAudioDataset(real_folder, fake_folder, transform, **melspect_params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader


def play(audio, sample_rate):
    with sd.OutputStream(
        channels=1,
        blocksize=2048,
        samplerate=sample_rate,
    ):
        sd.play(audio.T, sample_rate)
        sd.sleep(int(audio.shape[1]/sample_rate)*1000)


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):

    import matplotlib.pyplot as plt
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show()


def change_audio_len(audio, s_rate, time_ms):

    n_samples = int(s_rate * time_ms / 1000.0)
    signal_length = audio.shape[1]
    if signal_length > n_samples:
        audio = audio[:, :n_samples]
    elif signal_length < n_samples:
        n_missing_samples = n_samples - signal_length
        padding = (0, n_missing_samples)
        audio = torch.nn.functional.pad(audio, padding)

    return audio