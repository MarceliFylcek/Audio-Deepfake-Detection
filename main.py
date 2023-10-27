from mel_spectrogram import Mel_Spectrogram
import torch
import os
import wandb
from models import CNNModel
import torch.optim as optim
from sklearn.metrics import classification_report
from dataset import FakeAudioDataset
from config import MODELS_DIR, TRAIN_DIR, VALID_DIR
from tqdm import tqdm
import train_options
from utils import get_dataloader

"""
Folder structure:
/TRAIN_DIR
    /REAL
        audio1
        audio2
        ...
    /FAKE
/VALID_DIR
    /REAL
    /FAKE
"""

if __name__ == "__main__":
    # Parse input arguments
    args = train_options.parse()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    pretrained_name = args.load_pretrained
    checkpoint_freq = args.checkpoint_freq
    model_name = args.name
    no_valid = args.no_valid
    wandb_disabled = args.disable_wandb

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # start a new wandb run to track this script
    if not wandb_disabled:
        wandb.init(
            # set the wandb project where this run will be logged
            project="audio-deepfake-detection",
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": "CNN",
                "dataset": "11labs",
                "epochs": n_epochs,
            },
            name=args.name
        )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get training and validation dataloader
    train_dataloader = get_dataloader(TRAIN_DIR, batch_size)
    valid_dataloader = get_dataloader(VALID_DIR, batch_size, shuffle=False)

    # Dataloader returns batch and vector of labels
    # batch = [batch_size, 1, height, width]
    batch, labels = next(iter(train_dataloader))
    # Batch is passed to the model

    # Create the model
    m = CNNModel(n_filters=25, input_shape=[batch.shape[2], batch.shape[3]]).to(device)

    # Pretrained model loading
    if pretrained_name is not None:
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

    # Main progress bar
    main_progress_bar = tqdm(range(n_epochs), desc="Training progress", position=0)

    # For epoch
    for epoch in main_progress_bar:
        # Update epochs progress bar
        main_progress_bar.set_postfix(Epoch=f"{epoch} / {n_epochs}")

        # Set the model to training mode
        m.train()

        loss_train = 0

        train_epoch_progress = tqdm(
            train_dataloader, f"Epoch {epoch} (Train)", leave=False
        )

        # Training loop, for every batch
        for batch_idx, (batch, labels) in enumerate(train_epoch_progress):
            # zero the gradient buffers
            optimizer.zero_grad()

            # Data to device
            batch = batch.to(device)
            labels = labels.to(device)

            # Get the output
            output = m(batch)

            # Calculate loss
            loss = criterion(output, labels)

            # Calculate the gradient
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update running loss
            loss_train += loss.item() / batch_size

            # Update description of the sub-progress bar
            train_epoch_progress.set_postfix(
                Loss=f"{loss_train / (batch_idx + 1):.4f}"
            )

        train_epoch_progress.close()

        # Average loss per batch
        loss_train /= len(train_dataloader)

        if not no_valid:
            # Set the model to evaluation mode
            m.eval()

            correct_prediction = 0
            total = 0
            loss_valid = 0

            all_predicted = []
            all_labels = []

            valid_epoch_progress = tqdm(
                valid_dataloader, f"Epoch {epoch} (Valid)", leave=False
            )

            # No gradient calculation
            with torch.no_grad():
                # Evaluation loop
                for batch_idx, (batch, labels) in enumerate(valid_epoch_progress):
                    batch = batch.to(device)
                    labels = labels.to(device)
                    output = m(batch)
                    loss = criterion(output, labels)
                    loss_valid += loss.item() / batch_size
                    predicted = torch.argmax(output, axis=1)
                    all_predicted.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total += labels.size(0)
                    correct_prediction += (predicted == labels).sum().item()

                    # Update description of the sub-progress bar
                    valid_epoch_progress.set_postfix(
                        Loss=f"{loss_valid / (batch_idx + 1):.4f}"
                    )

            loss_valid /= len(valid_dataloader)
            accuracy = 100 * correct_prediction / total
            print(f"Accuracy: {accuracy}%")

            # Zero division error
            report = classification_report(
                all_predicted, all_labels, output_dict=True, zero_division=0
            )

            if not wandb_disabled:
                for class_name in list(report.keys())[:-3]:
                    for metric in list(report[class_name].keys())[:-1]:
                        wandb.log(
                            {f"{class_name}_{metric}": report[class_name][metric]}
                        )
                        # print(f'{class_name}_{metric}: {report[class_name][metric]}')

                wandb.log(
                    {
                        "loss_train": loss_train,
                        "loss_valid": loss_valid,
                        "accuracy": accuracy,
                    }
                )

        # Save the model
        if epoch % checkpoint_freq == 0:
            try:
                path = (
                        os.path.join(MODELS_DIR, model_name + "_e" + str(epoch))
                        + ".pth"
                )
                torch.save(m.state_dict(), path)
            except Exception as e:
                print(f"Error while saving the model: {str(e)}")
                quit()
            # print(f"Model {model_name} saved at epoch {epoch}")
