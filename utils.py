import os
from dataset import FakeAudioDataset
from torch.utils.data import DataLoader
import traceback

def get_dataloader(dataset_path, batch_size, melspect_params, shuffle=True,):
    real_folder = os.path.join(dataset_path, "real")
    fake_folder = os.path.join(dataset_path, "fake")

    #! Cuts only first 4 seconds of the recording as of now
    dataset = FakeAudioDataset(real_folder, fake_folder, **melspect_params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader