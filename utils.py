import os
from typing import Dict, Tuple
from dataset import FakeAudioDataset, SpeechIdentificationDataset
from torch.utils.data import DataLoader
import traceback
import sounddevice as sd
import torch
from torchvision.transforms import Normalize
import torch.nn as nn

class MinMaxNormalization(nn.Module):

    def __init__(self, org_min, org_max, new_min=0, new_max=1):
        super(MinMaxNormalization, self).__init__()

        self.org_min = org_min
        self.org_max = org_max

        self.new_min = new_min
        self.new_max = new_max

    def forward(self, x):

        x = ((x - self.org_min)/(self.org_max - self.org_min)) * (self.new_max - self.new_min) - self.new_min

        return x

def get_dataloader(dataset_path, batch_size, melspect_params, transform, normalize, shuffle=True, normalizer=None):
    real_folder = os.path.join(dataset_path, "real")
    fake_folder = os.path.join(dataset_path, "fake")

    if normalizer is None and normalize is not None:
        #! Cuts only first 4 seconds of the recording as of now
        dataset = FakeAudioDataset(real_folder, fake_folder, transform, None, **melspect_params)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
        if normalize == 'global_minmax':
            min_, max_ = get_min_max(dataloader)
            normalizer = MinMaxNormalization(min_, max_)
        elif normalize == 'global_std':
            mean, std = get_mean_std(dataloader)
            normalizer = Normalize(mean, std)
    
        dataset = FakeAudioDataset(real_folder, fake_folder, transform, normalizer, **melspect_params)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
    elif normalize is not None:
        # if we have normalizer ready for example with values obtained from training dataset
        dataset = FakeAudioDataset(real_folder, fake_folder, transform, normalizer, **melspect_params)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
    
    return dataloader, normalizer


def get_speakers_dataloader(
    speakers_path, batch_size, melspect_params, transform, shuffle=True
):
    dataset = SpeechIdentificationDataset(speakers_path, transform, **melspect_params)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return dataloader


def play(audio, sample_rate):
    with sd.OutputStream(
        channels=1,
        blocksize=2048,
        samplerate=sample_rate,
    ):
        sd.play(audio.T, sample_rate)
        sd.sleep(int(audio.shape[1] / sample_rate) * 1000)


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


def normalize_batch(batch):
    batch_m, batch_s = batch.mean(), batch.std()
    batch = (batch - batch_m) / batch_s
    return batch


def get_mean_std(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:

    fst_moment = 0.0
    snd_moment = 0.0
    cnt = 0

    for batch, _ in dataloader:
        b, c, h, w = batch.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(batch, dim=[0, 2, 3])
        sum_of_square = torch.sum(batch**2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)

    return mean, std

def get_min_max(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val = torch.inf
    max_val = -torch.inf

    for batch, _ in dataloader:
        b_max = torch.max(batch)
        b_min = torch.min(batch)

        if b_max > max_val:
            max_val = b_max
        if b_min < min_val:
            min_val = b_min
    
    return min_val, max_val


def get_librispeech_names(file_path: str) -> Dict[int, str]:
    """
    Returns a dictionary with speaker id and speaker name from librispeech
    dataset.

    Args:
        file_path (str)

    Returns:
        Dict[int:str]
    """

    speaker_names = {}

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if not line.startswith(";"):
            fields = line.strip().split("|")
            speaker_id = int(fields[0].strip())
            name = fields[-1].strip()
            speaker_names[speaker_id] = name

    return speaker_names

