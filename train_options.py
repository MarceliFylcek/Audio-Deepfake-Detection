import argparse


def parse():
    parser = argparse.ArgumentParser(description="Set training options.")
    parser.add_argument("--name", default="model", type=str, help="Name for the new model")
    parser.add_argument("--n_epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--checkpoint_freq", default=1,type=int, help="How many epochs between saving the model's checkpoint",)
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--load_pretrained", default=None, type=str, help="Name of a pretrained model")
    parser.add_argument("--no", default=None, type=str, help="Name of a pretrained model")
    parser.add_argument("--no_valid", action="store_true", help="No validation set")
    parser.add_argument("--disable_wandb", default=False, action="store_true", help="Whether to disable wandb or not")
    parser.add_argument("--normalization", default="global_std", choices=["none", "batch", "global_minmax", "global_std"], help="Specify type of normalization")

    args = parser.parse_args()
    return args
