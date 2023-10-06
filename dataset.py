from typing import Tuple
import torch
from torch.utils.data import Dataset
import os
from Mel_Spectrogram import Mel_Spectrogram

class FakeAudioDataset(Dataset):
    def __init__(self, real_folder: str, fake_folder: str,
                time_milliseconds: int, sampling_rate, n_mels):
        """Spoof audio dataset

        Args:
            real_folder (str): path to folder with real audio files
            fake_folder (str): path to folder with fake audio files
            n_samples (int): maximum number of samples for a file
            sampling_rate (int): desired sampling rate
        """
        self.real_folder = real_folder
        self.fake_folder = fake_folder
        self.time_milliseconds = time_milliseconds
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels

        # Path and label
        real_paths = [[os.path.join(self.real_folder,
                    filename)] for filename in os.listdir(self.real_folder)]
        

        fake_paths = [[os.path.join(self.fake_folder,
                    filename)] for filename in os.listdir(self.fake_folder)]
        
        self.filepaths = real_paths + fake_paths

        # 1 for real audio, 0 for fake audio
        self.labels = [1 for _ in real_paths] + [0 for _ in fake_paths]


    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        
        path = self.filepaths[index][0]
        label = self.labels[index]

        spectogram = Mel_Spectrogram(path, self.sampling_rate,
                                     self.n_mels, self.time_milliseconds)

        raw_data = spectogram.get_raw_data()

        return raw_data, label



