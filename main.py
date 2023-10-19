from Mel_Spectrogram import Mel_Spectrogram
from torch.utils.data import DataLoader
import torch
import os
import wandb
from models import CNNModel
import torch.optim as optim
from sklearn.metrics import classification_report

# # # start a new wandb run to track this script
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

d_sample, _ =  next(iter(dataloader))
m = CNNModel(16, torch.unsqueeze(d_sample, 1).shape[2:])
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(m.parameters(), lr=0.001)
for epoch in range(10):
    optimizer.zero_grad()   # zero the gradient buffers
    for sample in dataloader:
        raw_data, label = sample
        inputs = torch.unsqueeze(raw_data, 1)
        output = m(inputs)
        target = torch.zeros(output.shape)
        target[torch.arange(0,label.shape[0]), label] = 1
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        report = classification_report(torch.argmax(output, 1), torch.argmax(target, 1), output_dict=True)

        for class_name in list(report.keys())[:-3]:
            for metric in list(report[class_name].keys())[:-1]:
                wandb.log({f'{class_name}_{metric}': report[class_name][metric]})
                print(f'{class_name}_{metric}: {report[class_name][metric]}')