from Mel_Spectrogram import Mel_Spectrogram
from torch.utils.data import DataLoader
import torch
import os
import wandb
from models import CNNModel
import torch.optim as optim

# # start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="audio-deepfake-detection",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "CNN",
    "dataset": "11labs",
    "epochs": 10,
    }
)


spectrogram = Mel_Spectrogram("resources/test.mp3", new_sample_rate=16_000,
                            n_mels=64, time_milliseconds=5_000)

spectrogram.display()
spectrogram.play()

from dataset import FakeAudioDataset

real_folder = os.path.join('resources', 'real')
fake_folder = os.path.join('resources', 'fake')

dataset = FakeAudioDataset(real_folder, fake_folder, 4_000, 16_000, 64)

dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)

for sample in dataloader:
    raw_data, label = sample
    print(raw_data.shape, label)
    

inputs = torch.unsqueeze(raw_data, 1)
m = CNNModel(16, inputs.shape[2:])
m.zero_grad()
out = m(inputs)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(m.parameters(), lr=0.01)
for epoch in range(10):
    optimizer.zero_grad()   # zero the gradient buffers
    output = m(inputs)
    target = torch.zeros(out.shape)
    target[torch.arange(0,label.shape[0]), label] = 1
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    wandb.log({"loss": loss})

print('a')