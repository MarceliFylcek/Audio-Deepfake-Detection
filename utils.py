import os
from dataset import FakeAudioDataset
from torch.utils.data import DataLoader

def get_dataloader(dataset_path, batch_size, shuffle=True):
    real_folder = os.path.join(dataset_path, "real")
    fake_folder = os.path.join(dataset_path, "fake")

    try:
        #! Cuts only first 4 seconds of the recording as of now
        dataset = FakeAudioDataset(real_folder, fake_folder, 4_000, 16_000, 64)
    except Exception as e:
        print(f"Error while creating the dataset: {str(e)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader