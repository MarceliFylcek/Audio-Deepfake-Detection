from Mel_Spectrogram import Mel_Spectrogram
from torch.utils.data import DataLoader
import os

spectrogram = Mel_Spectrogram("resources/test.mp3", new_sample_rate=16_000,
                            n_mels=64, time_milliseconds=5_000)

#spectrogram.display()
spectrogram.play()

from dataset import FakeAudioDataset

real_folder = os.path.join('resources', 'real')
fake_folder = os.path.join('resources', 'fake')

dataset = FakeAudioDataset(real_folder, fake_folder, 4_000, 16_000, 64)

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=0)

for sample in dataloader:
    raw_data, label = sample
    print(raw_data.shape, label)
    