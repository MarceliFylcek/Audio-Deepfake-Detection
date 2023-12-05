from typing import Tuple
import torch
from torch.utils.data import Dataset
import os
from transforms.mel_spectrogram import Mel_Spectrogram
from transforms.mfcc import MFCC
from transforms.spectrogram import Spectrogram
import random
from augmentation import room_reverb, add_noise

class FakeAudioDataset(Dataset):
    def __init__(
        self,
        real_folder: str,
        fake_folder: str,
        transform,
        augmentations: bool,
        normalize,
        time_milliseconds: int,
        new_sample_rate,
        n_bins,
        db_amplitude,
    ):
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
        self.sampling_rate = new_sample_rate
        self.n_bins = n_bins
        self.db_amplitude = db_amplitude
        self.transform = transform
        self.augmentations = augmentations
        self.normalize = normalize

        # Path and label
        real_paths = [
            [os.path.join(self.real_folder, filename)]
            for filename in os.listdir(self.real_folder)
        ]

        fake_paths = [
            [os.path.join(self.fake_folder, filename)]
            for filename in os.listdir(self.fake_folder)
        ]

        self.filepaths = real_paths + fake_paths

        # 1 for real audio, 0 for fake audio
        self.labels = [1 for _ in real_paths] + [0 for _ in fake_paths]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        path = self.filepaths[index][0]
        label = self.labels[index]

        transform = self.transform(
            path,
            self.sampling_rate,
            self.n_bins,
            self.time_milliseconds,
            self.db_amplitude,
        )
        if self.normalize is not None:
            # normalization requires data to be of shape (..., C, H, W)
            raw_data = self.normalize(transform.get_raw_data().reshape((1, 1)+transform.get_raw_data().shape))
            raw_data = raw_data.squeeze(dim=0)
        else:
            raw_data = transform.get_raw_data()
            # Add 1 for channels
            raw_data = raw_data.unsqueeze(dim=0)

        return raw_data, label


class SpeechIdentificationDataset(Dataset):
    def __init__(
        self,
        speakers_dir: str,
        transform,
        time_milliseconds: int,
        new_sample_rate,
        n_bins,
        db_amplitude,
    ):
        """Speech indentification dataset"""

        self.speakers_dir = speakers_dir
        self.time_milliseconds = time_milliseconds
        self.sampling_rate = new_sample_rate
        self.n_bins = n_bins
        self.db_amplitude = db_amplitude
        self.transform = transform

        self.file_paths = []
        self.labels = []

        for speaker_id in os.listdir(self.speakers_dir):
            for utterance in speaker_id:
                self.file_paths.extend(
                    os.path.join(self.speakers_dir, speaker_id, utterance)
                )
                self.labels.extend(speaker_id)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.recording_paths[index]
        label = self.labels[index]

        transform = self.transform(
            path,
            self.sampling_rate,
            self.n_bins,
            self.time_milliseconds,
            self.db_amplitude,
        )


        raw_data = transform.get_raw_data()

        ### AUGMENTACJE ###
        if self.augmentations:
            random_number = random.randrange(5)
            if random_number == 0:
                raw_data = room_reverb(raw_data, self.sampling_rate)
            elif random_number == 1:
                raw_data = add_noise(raw_data, self.sampling_rate)

        # Add 1 for channels
        raw_data = raw_data.unsqueeze(dim=0)

        return raw_data, label
