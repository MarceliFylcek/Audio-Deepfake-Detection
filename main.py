from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from transformers import Dinov2Config

from transforms.mel_spectrogram import Mel_Spectrogram
from transforms.mfcc import MFCC
from transforms.spectrogram import Spectrogram
from transforms.cqt import CQT
import torch
import os
import wandb
from models import CNNModel, DinoV2TransformerBasedModel, CNN_LSTM_Model, get_VIT
import torch.optim as optim
from sklearn.metrics import classification_report
from config import MODELS_DIR, TRAIN_DIR, VALID_DIR, TRAIN_DIR_11LABS, VALID_DIR_11LABS, TRAIN_DIR_COREJ, VALID_DIR_COREJ, TRAIN_DIR_MIXED, VALID_DIR_MIXED, melspectogram_params, melspectogram_params_vit16

from tqdm import tqdm
import train_options
from utils import get_dataloader, normalize_batch

# melspectogram_params = melspectogram_params_vit16 #for pretrained transformer

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

### dataset = [TRAIN_DIR, VALID_DIR]
### name = "Mel_Spectrogram", "MFCC", "Spectrogram", "CQT"
### transform = Mel_Spectrogram, MFCC, Spectrogram, CQT
def train(architecture, dataset, transformer, augmentation, transformer_name):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        # Parse input arguments
        args = train_options.parse()

        n_epochs = args.n_epochs
        batch_size = args.batch_size
        lr = args.lr
        pretrained_name = args.load_pretrained
        checkpoint_freq = args.checkpoint_freq
        no_valid = args.no_valid
        wandb_disabled = args.disable_wandb
        normalization = args.normalization

        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        # start a new wandb run to track this script
        if not wandb_disabled:
            wandb.init(
                project="audio-deepfake-tests",
                config={
                    "learning_rate": lr,
                    "architecture": architecture,
                    "dataset": f"{dataset[0]}_{dataset[1]}",
                    "epochs": n_epochs,
                },
                name=f"{architecture}_{dataset[0].replace('/', '_')}_{dataset[1].replace('/', '_')}_{transformer_name}_augmentation={augmentation}"
            )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Get training and validation dataloader
        train_dataloader, normalizer = get_dataloader(
            TRAIN_DIR, batch_size, melspect_params=melspectogram_params, transform=transformer, normalize=normalization, augmentation=augmentation
        )
        valid_dataloader, _ = get_dataloader(
            VALID_DIR,
            batch_size,
            shuffle=False,
            melspect_params=melspectogram_params,
            transform=Spectrogram,
            normalize=normalization,
            normalizer=normalizer
        )

        batch, labels = next(iter(train_dataloader))

        ### ARCHITECTURE ###
        if architecture == "CNN":
            model = CNNModel(n_filters=5, input_shape=[batch.shape[2], batch.shape[3]])
        elif architecture == "CNN+LSTM":
            model = CNN_LSTM_Model(n_filters=25, input_shape=[batch.shape[2], batch.shape[3]], hidden_size=1024, num_layers=batch.shape[3])
        elif architecture == "Transformer":
            config = Dinov2Config(num_channels=1, patch_size=4, hidden_size=48)
            model = DinoV2TransformerBasedModel(config, train_dataloader.dataset[0][0].shape[-2:])
        model.to(device)

        ### LOAD PRETRAINED IF PROVIDED ###
        if pretrained_name is not None:
            try:
                path = os.path.join(MODELS_DIR, pretrained_name) + ".pth"
                model.load_state_dict(torch.load(path))
            except Exception as e:
                print(f"Error while loading the model: {str(e)}")
                quit()
            print(f"Model {pretrained_name} loaded successfully")

        ### OPTIMIZER ###
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=int(1.3 * n_epochs), eta_min=0.000001
        )
        main_progress_bar = tqdm(
            range(n_epochs), desc="Training progress", position=0
        )

        ### EPOCHS ###
        for epoch in main_progress_bar:
            main_progress_bar.set_postfix(Epoch=f"{epoch} / {n_epochs}")
            model.train()
            loss_train = 0
            train_epoch_progress = tqdm(
                train_dataloader, f"Epoch {epoch} (Train)", leave=False
            )
            step_counter = 0

            for batch_idx, (batch, labels) in enumerate(train_epoch_progress):
                step_counter += 1
                optimizer.zero_grad()

                # Data to device
                batch = batch.to(device)
                labels = labels.to(device)

                #Batch normalization (no learnable params)
                if normalization == 'batch':
                    batch = normalize_batch(batch)
                elif normalization == "global_minmax" or normalization == "global_std":
                    batch = normalizer(batch)

                output = model(batch)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step(epoch + int(step_counter / 192))
                loss_train += loss.item() / batch_size

                # Update description of the sub-progress bar
                train_epoch_progress.set_postfix(
                    Loss=f"{loss_train / (batch_idx + 1):.4f}",
                    lr=optimizer.param_groups[0]["lr"],
                )

            train_epoch_progress.close()
            loss_train /= len(train_dataloader)

            ### VALIDATION ###
            if not no_valid:
                model.eval()

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
                    for batch_idx, (batch, labels) in enumerate(
                        valid_epoch_progress
                    ):
                        batch = batch.to(device)
                        labels = labels.to(device)
                        if normalization == 'batch':
                            batch = normalize_batch(batch)
                        elif normalization == "global_minmax" or normalization == "global_std":
                            batch = normalizer(batch)
                        output = model(batch)
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
                print(f"\nAccuracy: {accuracy}%")

                # Zero division error
                report = classification_report(
                    all_predicted, all_labels, output_dict=True, zero_division=0
                )

                if not wandb_disabled:
                    for class_name in list(report.keys())[:-3]:
                        for metric in list(report[class_name].keys())[:-1]:
                            wandb.log(
                                {
                                    f"{class_name}_{metric}": report[class_name][
                                        metric
                                    ]
                                }
                            )
                    wandb.log(
                        {
                            "loss_train": loss_train,
                            "loss_valid": loss_valid,
                            "accuracy": accuracy,
                        }
                    )

            ### SAVE MODEL ###
            if epoch % checkpoint_freq == 0:
                try:
                    path = (
                        os.path.join(MODELS_DIR, f"{architecture}_{dataset[0].replace('/', '_')}_{dataset[1].replace('/', '_')}_{transformer_name}_augmentation={augmentation}_e" + str(epoch))
                        + ".pth"
                    )
                    print(path)
                    torch.save(model.state_dict(), path)
                except Exception as e:
                    print(f"Error while saving the model: {str(e)}")
                    quit()
                print(f"Model {architecture}_{dataset[0].replace('/', '_')}_{dataset[1].replace('/', '_')}_{transformer_name}_augmentation={augmentation}_e + str(epoch) saved at epoch {epoch}")

        main_progress_bar.close()    

        wandb.finish()


if __name__ == "__main__":
    ### EXAMPLE ###
    '''
    (function) def train(
    architecture: Any,
    dataset: Any,
    transformer: Any,
    augmentation: Any,
    transformer_name: Any
        ) -> Any
    '''
    train("CNN", [TRAIN_DIR_11LABS, VALID_DIR_11LABS], Mel_Spectrogram, False, "Mel_Spectrogram")
    

