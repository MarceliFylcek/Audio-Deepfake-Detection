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
from config import MODELS_DIR, DATASET_DIR
from tqdm import tqdm


if __name__ == "__main__":

    # Parse input arguments
    #! Default parameters are set here
    parser = argparse.ArgumentParser(description="Set training options.")
    parser.add_argument("--name", default="model", type=str, help="Name for the new model")
    parser.add_argument("--n_epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--checkpoint_freq",default=1, type=int, help="How many epochs between saving the model checkpoint")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--load_pretrained", default=None, type=str, help="Name of a pretrained model")

    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    pretrained_name = args.load_pretrained
    checkpoint_freq = args.checkpoint_freq
    model_name = args.name

    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="audio-deepfake-detection",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": lr,
    #     "architecture": "CNN",
    #     "dataset": "11labs",
    #     "epochs": n_epochs,
    #     }
    # )

    # Set paths to data
    real_folder = os.path.join(DATASET_DIR, 'real')
    fake_folder = os.path.join('resources', 'fake')

    # Create a dataset
    try:
        #! Cuts only first 4 seconds of the recording as of now
        dataset = FakeAudioDataset(real_folder, fake_folder, 4_000, 16_000, 64)
    except Exception as e:
        print(f"Error creating the dataset: {str(e)}")

    # Create 
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    # Dataloader returns batch and vector of labels
    # batch = [batch_size, height, width]
    batch, labels =  next(iter(dataloader))
    # Batch is passed to the model

    # Create the model
    m = CNNModel(n_filters=25, input_shape=[batch.shape[2], batch.shape[3]])

    # Model loading
    if pretrained_name != None:
        try:
           path = os.path.join(MODELS_DIR, pretrained_name) + ".pth"
           m.load_state_dict(torch.load(path))
        except Exception as e:
            print(f"Error while loading the model: {str(e)}")
            quit()
        print(f"Model {pretrained_name} loaded successfully")

    # Loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(m.parameters(), lr=lr)

    # For epoch
    for epoch in range(n_epochs):
        loop = tqdm(dataloader)

        # For batch
        for idx, (x, y) in enumerate(loop):

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

            # Progress bar description
            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(loss=loss.item())

            # Zero division error
            report = classification_report(torch.argmax(output, axis=1), labels, output_dict=True, zero_division=0)

            # for class_name in list(report.keys())[:-3]:
            #     for metric in list(report[class_name].keys())[:-1]:
            #         wandb.log({f'{class_name}_{metric}': report[class_name][metric]})
            #         # print(f'{class_name}_{metric}: {report[class_name][metric]}')

        # Save the model
        if epoch % checkpoint_freq == 0:
            try:
                path = os.path.join(MODELS_DIR, model_name + "_e" + str(epoch)) + ".pth"
                torch.save(m.state_dict(), path)
            except Exception as e:
                print(f"Error while saving the model: {str(e)}")
                quit()
            print(f"Model {model_name} saved at epoch {epoch}")