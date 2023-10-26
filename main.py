from Mel_Spectrogram import Mel_Spectrogram
from torch.utils.data import DataLoader
import torch
import os
import wandb
from models import CNNModel
import torch.optim as optim
from sklearn.metrics import classification_report
from dataset import FakeAudioDataset
import argparse


if __name__ == "__main__":

    # Parse input arguments
    #! Default parameters are set here
    parser = argparse.ArgumentParser(description="Set training options.")
    parser.add_argument("--n_epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")

    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="audio-deepfake-detection",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "CNN",
        "dataset": "11labs",
        "epochs": n_epochs,
        }
    )

    # Set paths to data
    real_folder = os.path.join('resources', 'real')
    fake_folder = os.path.join('resources', 'fake')

    # Create a dataset
    dataset = FakeAudioDataset(real_folder, fake_folder, 4_000, 16_000, 64)

    # Create 
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    # Dataloader returns batch and vector of labels
    # batch = [batch_size, height, width]
    batch, labels =  next(iter(dataloader))
    # Batch is passed to the model

    # Create the model
    m = CNNModel(n_filters=25, input_shape=[batch.shape[2], batch.shape[3]])

    # Loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(m.parameters(), lr=lr)

    # For epoch
    for epoch in range(n_epochs):
        # For batch
        for sample in dataloader:

            # zero the gradient buffers
            optimizer.zero_grad()

            # Get the batch and label vector
            batch, labels =  next(iter(dataloader))

            # Get the output
            output = m(batch)

            # Calculate loss
            loss = criterion(output, labels)

            # Calculate the gradient
            loss.backward()

            # Update the weights
            optimizer.step()

            report = classification_report(torch.argmax(output, axis=1), labels, output_dict=True)

            for class_name in list(report.keys())[:-3]:
                for metric in list(report[class_name].keys())[:-1]:
                    wandb.log({f'{class_name}_{metric}': report[class_name][metric]})
                    print(f'{class_name}_{metric}: {report[class_name][metric]}')