import torch
import os
from transforms.mfcc import MFCC
from sklearn.metrics import classification_report
from utils import get_speakers_dataloader
from config import melspectogram_params
import argparse
import train_options
import wandb
import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from utils import normalize_batch


MODELS_DIR = "models"
EMBEDDING_LENGTH = 512

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
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
    normalization = args.normalization

    if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    # start a new wandb run to track this script
    # if not wandb_disabled:
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="audio-deepfake-detection",
    #         # track hyperparameters and run metadata
    #         config={
    #             "learning_rate": lr,
    #             "epochs": n_epochs,
    #         },
    #         name=args.name
    #     )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Speakers training dataloader        
    dataloader = get_speakers_dataloader(
        "test-clean", 16, melspectogram_params, MFCC, True
    )

    # Dataloader returns batch and vector of labels
    # batch = [batch_size, 1, height, width]
    batch, labels = next(iter(dataloader))

    # Create the model
    #! Not finished yet
    # m = ResCNN()

    # Pretrained model loading
    if pretrained_name is not None:
        try:
            path = os.path.join(MODELS_DIR, pretrained_name) + ".pth"
            m.load_state_dict(torch.load(path))
        except Exception as e:
            print(f"Error while loading the model: {str(e)}")
            quit()
        print(f"Model {pretrained_name} loaded successfully")

    #! Criterion

    # Optimizer
    optimizer = optim.Adam(m.parameters(), lr=lr)

    # Runtime learning rate modifier
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=int(1.3 * n_epochs), eta_min=0.000001
    )

    # Main progress bar
    main_progress_bar = tqdm(
        range(n_epochs), desc="Training progress", position=0
    )

    # For epoch
    for epoch in main_progress_bar:
        # Update epochs progress bar
        main_progress_bar.set_postfix(Epoch=f"{epoch} / {n_epochs}")

        # Set the model to training mode
        m.train()

        loss_train = 0

        train_epoch_progress = tqdm(
            dataloader, f"Epoch {epoch} (Train)", leave=False
        )

        # Set step counter for learning rate scheduler
        step_counter = 0
        # Training loop, for every batch
        for batch_idx, (batch, labels) in enumerate(train_epoch_progress):
            step_counter += 1
            # zero the gradient buffers
            optimizer.zero_grad()

            # Data to device
            batch = batch.to(device)
            labels = labels.to(device)

            #! Batch normalization (no learnable params)
            if normalization:
                batch = normalize_batch(batch)

            # Get the output
            output = m(batch)

            # Calculate loss
            loss = criterion(output, labels)

            # Calculate the gradient
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update learning rate
            lr_scheduler.step(epoch + int(step_counter / 192))

            # Update running loss
            loss_train += loss.item() / batch_size

            # Update description of the sub-progress bar
            train_epoch_progress.set_postfix(
                Loss=f"{loss_train / (batch_idx + 1):.4f}",
                lr=optimizer.param_groups[0]["lr"],
            )

        train_epoch_progress.close()

        # Average loss per batch
        loss_train /= len(dataloader)

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
