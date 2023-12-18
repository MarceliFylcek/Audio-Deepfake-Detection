from main import train
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
from config import MODELS_DIR, TRAIN_DIR, VALID_DIR, TRAIN_DIR_11LABS, VALID_DIR_11LABS, TRAIN_DIR_COREJ, VALID_DIR_COREJ, TRAIN_DIR_MIXED, VALID_DIR_MIXED,  melspectogram_params, melspectogram_params_vit16

from tqdm import tqdm
import train_options
from utils import get_dataloader, normalize_batch

### OPCJE ###
ARCHITECTURES = ["CNN", "CNN+LSTM", "Transformer"]
TRAIN_DATASETS = [TRAIN_DIR_11LABS, TRAIN_DIR_COREJ, TRAIN_DIR_MIXED]
VALID_DATASETS = [VALID_DIR_11LABS, VALID_DIR_COREJ, VALID_DIR_MIXED]
TRAIN_OPTIONS = [Mel_Spectrogram, MFCC, Spectrogram, CQT]
TRAIN_OPTIONS_LABELS = ["Mel_Spectrogram", "MFCC", "Spectrogram", "CQT"]
AUGMENTATIONS = [False, True]

for architecture in ARCHITECTURES:
        for dataset in TRAIN_DATASETS:
            for train_index, train_option in enumerate(TRAIN_OPTIONS):
                for augmentation in AUGMENTATIONS:
                     train(architecture, dataset, VALID_DATASETS, train_option, augmentation,
                           transformer_name=TRAIN_OPTIONS_LABELS[train_index])