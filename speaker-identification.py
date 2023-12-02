import torch
import os
from transforms.mfcc import MFCC
from sklearn.metrics import classification_report
from utils import get_speakers_dataloader
from config import melspectogram_params

if __name__ == "__main__":

    dataloader = get_speakers_dataloader(
        "test-clean", 16, melspectogram_params, MFCC, True
    )
    EMBEDDING_LENGTH = 512

    # model = ResCNN()
