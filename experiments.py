import os
import train_options
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from transformers import Dinov2Config

from transforms.mel_spectrogram import Mel_Spectrogram
from transforms.mfcc import MFCC
from transforms.spectrogram import Spectrogram
from transforms.cqt import CQT

import torch
import os
import wandb
from models import CNNModel, DinoV2TransformerBasedModel, CNN_LSTM_Model
import torch.optim as optim
from sklearn.metrics import classification_report
from config import MODELS_DIR, TRAIN_DIR_11LABS, VALID_DIR_11LABS, TRAIN_DIR_COREJ, VALID_DIR_COREJ, TRAIN_DIR_MIXED, VALID_DIR_MIXED,melspectogram_params
from tqdm import tqdm
from utils import get_dataloader, normalize_batch
from augmentation import reverb_audio, noisy_audio, speech_audio
import random


### OPCJE ###
ARCHITECTURES = ["CNN", "CNN+LSTM", "Transformer"]
DATESETS = [[TRAIN_DIR_11LABS, VALID_DIR_11LABS],
           [TRAIN_DIR_11LABS, VALID_DIR_COREJ],
           [TRAIN_DIR_COREJ, VALID_DIR_COREJ],
           [TRAIN_DIR_COREJ, VALID_DIR_11LABS],
           [TRAIN_DIR_MIXED, VALID_DIR_MIXED]]
TRAIN_OPTIONS = [Mel_Spectrogram, MFCC, Spectrogram, CQT]
AUGMENTATIONS = [True, False]

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    args = train_options.parse()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    pretrained_name = args.load_pretrained
    checkpoint_freq = args.checkpoint_freq
    model_name = args.name
    no_valid = args.no_valid
    wandb_disabled = args.disable_wandb
    normalization = args.normalization

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    for architecture in ARCHITECTURES:
        for dataset in DATESETS:
            for train_option in TRAIN_OPTIONS:
                for augmentation in AUGMENTATIONS:
                    if not wandb_disabled:
                        wandb.init(
                            project="audio-deepfake-tests",
                            config={
                                "learning_rate": lr,
                                "architecture": architecture,
                                "dataset": f"{dataset[0]}_{dataset[1]}",
                                "epochs": n_epochs,
                            },
                            name=f"{architecture}_{dataset[0]}_{dataset[1]}_{train_option}_{augmentation}"
                        )

                    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                    print(f"Using device: {device}")

                    ### DATA + TRAIN_OPTION ###
                    train_dataloader = get_dataloader(
                        dataset[0], batch_size, melspect_params=melspectogram_params, transform=train_option
                    )
                    valid_dataloader = get_dataloader(
                        dataset[1],
                        batch_size,
                        shuffle=False,
                        melspect_params=melspectogram_params,
                        transform=train_option
                    )

                    ### AUGMENTATIONS ###
                    if augmentation:
                        for index in range(train_dataloader):
                            random_number = random.randrange(3)
                            if random_number == 0:
                                batch[index] = reverb_audio(train_dataloader.dataset[index])
                            elif random_number == 1:
                                batch[index] = noisy_audio(train_dataloader.dataset[index])
                            elif random_number == 2:
                                batch[index] = speech_audio(train_dataloader.dataset[index])
                        for index in range(valid_dataloader):
                            random_number = random.randrange(3)
                            if random_number == 0:
                                batch[index] = reverb_audio(valid_dataloader.dataset[index])
                            elif random_number == 1:
                                batch[index] = noisy_audio(valid_dataloader.dataset[index])
                            elif random_number == 2:
                                batch[index] = speech_audio(valid_dataloader.dataset[index])


                    batch, labels = next(iter(train_dataloader))

                    ### ARCHITECTURE ###
                    if architecture == "CNN":
                        model = CNNModel(n_filters=5, input_shape=[batch.shape[2], batch.shape[3]]).to(device)
                    elif architecture == "CNN+LSTM":
                        model = CNN_LSTM_Model(n_filters=25, input_shape=[batch.shape[2], batch.shape[3]], hidden_size=1024, num_layers=batch.shape[3]).to(device)
                    elif architecture == "Transformer":
                        config = Dinov2Config(num_channels=1, patch_size=4, hidden_size=48)
                        model = DinoV2TransformerBasedModel(config, train_dataloader.dataset[0][0].shape[-2:]).to(device)

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
                            batch = batch.to(device)
                            labels = labels.to(device)
                            if normalization:
                                batch = normalize_batch(batch)
                            output = model(batch)
                            loss = criterion(output, labels)
                            loss.backward()
                            optimizer.step()
                            lr_scheduler.step(epoch + int(step_counter / 192))
                            loss_train += loss.item() / batch_size
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
                            with torch.no_grad():
                                # Evaluation loop
                                for batch_idx, (batch, labels) in enumerate(
                                    valid_epoch_progress
                                ):
                                    batch = batch.to(device)
                                    labels = labels.to(device)
                                    if normalization:
                                        batch = normalize_batch(batch)
                                    output = model(batch)
                                    loss = criterion(output, labels)
                                    loss_valid += loss.item() / batch_size
                                    predicted = torch.argmax(output, axis=1)
                                    all_predicted.extend(predicted.cpu().numpy())
                                    all_labels.extend(labels.cpu().numpy())
                                    total += labels.size(0)
                                    correct_prediction += (predicted == labels).sum().item()
                                    valid_epoch_progress.set_postfix(
                                        Loss=f"{loss_valid / (batch_idx + 1):.4f}"
                                    )
                            loss_valid /= len(valid_dataloader)
                            accuracy = 100 * correct_prediction / total
                            print(f"\nAccuracy: {accuracy}%")
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
                                    os.path.join(MODELS_DIR, f"{architecture}_{dataset[0]}_{dataset[1]}_{train_option}_{augmentation}_e" + str(epoch))
                                    + ".pth"
                                )
                                torch.save(model.state_dict(), path)
                            except Exception as e:
                                print(f"Error while saving the model: {str(e)}")
                                quit()
                            print(f"Model {architecture}_{dataset[0]}_{dataset[1]}_{train_option}_{augmentation}_e + str(epoch) saved at epoch {epoch}")

                    main_progress_bar.close()    