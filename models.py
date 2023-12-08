from abc import ABC
from typing import Tuple

import torch.nn as nn
import torch
from torch import Tensor
from transformers import Dinov2PreTrainedModel, Dinov2Model
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights


def calc_shape(in_shape, padding, dilation, k_size, stride):
    h = (in_shape[0] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1
    w = (in_shape[1] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1
    return (int(h), int(w))


class CNNModel(nn.Module):
    def __init__(self, n_filters, input_shape):
        """ """
        super(CNNModel, self).__init__()

        # In one channel, out n_filters
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        # Calculate next layer input size
        conv_out = calc_shape([input_shape[0], input_shape[1]], 0, 1, 3, 1)
        maxp_out = calc_shape([conv_out[0], conv_out[1]], 0, 1, 2, 1)

        self.fc1 = nn.Linear(maxp_out[0] * maxp_out[1] * n_filters, 256)
        self.fc2 = nn.Linear(256, 2)

        # Not needed
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc2(x)

        # Not needed
        # x = self.softmax(x)

        return x
    

class CNN_LSTM_Model(nn.Module):
    """CNN with LSTM

    Input shape of LSTM layers: [batch_size, sequence_len, input_size]
    All intermediate hidden states are passed to linear layers.
    Input size of dense layers: [num_layers * hidden_size]

    """

    def __init__(self, n_filters, input_shape, hidden_size, num_layers):
        super(CNN_LSTM_Model, self).__init__()

        self.n_filters = n_filters
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        conv_out = calc_shape([input_shape[0], input_shape[1]], 0, 1, 3, 1)
        self.maxp_out = calc_shape([conv_out[0], conv_out[1]], 0, 1, 2, 1)

        self.lstm1 = nn.LSTM(
            input_size= n_filters * self.maxp_out[0],
            hidden_size=hidden_size,
            num_layers=self.maxp_out[1],
        )
        self.fc1 = nn.Linear(self.maxp_out[1]* hidden_size, 256)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Reshape for LSTM -> [batch, sequence_len, data]
        x = x.view(x.size(0), x.size(3), -1)

        x, _ = self.lstm1(x)

        # Flatten the output for the Linear layer
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class DenseClassifier(nn.Module):
    def __init__(self, hidden_size: int, input_size: int):
        super(DenseClassifier, self).__init__()
        self.Dense_A = nn.Linear(input_size * hidden_size, input_size * hidden_size // 8)
        self.Dense_B = nn.Linear(input_size * hidden_size // 8, input_size * hidden_size // 24)
        self.Output_Dense = nn.Linear(input_size * hidden_size // 24, 2)
        self.relu = nn.ReLU()

        # self.Dense_A = nn.Linear(496 * hidden_features, 256 * hidden_features)
        # self.Dense_B = nn.Linear(256 * hidden_features, 128 * hidden_features)
        # self.Output_Dense = nn.Linear(128 * hidden_features, 2)
        # self.relu = nn.ReLU()

    def forward(self, patch_embeddings: Tensor):
        x = self.relu(self.Dense_A(patch_embeddings.reshape(patch_embeddings.shape[0], -1)))
        x = self.relu(self.Dense_B(x))
        x = self.Output_Dense(x)
        return x


class DinoV2TransformerBasedModel(Dinov2PreTrainedModel, ABC):
    def __init__(self, dinoV2_config, input_shape: Tuple[int, int]):
        super().__init__(dinoV2_config)
        self.dinoV2 = Dinov2Model(dinoV2_config)
        self.patch_size = dinoV2_config.patch_size
        self.hidden_size = dinoV2_config.hidden_size

        self.input_shape = input_shape
        self.n_output_features = self.get_n_output_features(input_shape)
        self.patches_shape = self.get_patches_shape(input_shape)

        self.classifier = DenseClassifier(self.hidden_size, self.n_output_features)

    def get_patches_shape(self, input_shape: Tuple[int, int]):
        return input_shape[0] // self.patch_size, input_shape[1] // self.patch_size

    def get_n_output_features(self, input_shape: Tuple[int, int]):
        return (input_shape[0] // self.patch_size) * (input_shape[1] // self.patch_size)

    def forward(self, spectrograms: Tensor):
        outputs = self.dinoV2(spectrograms, output_attentions=False)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        logits = self.classifier(patch_embeddings)
        return logits

class MLP(nn.Module):
    """MLP model to append to transformer

    Args:
        MLP model
    """
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):

        x = self.fc1(x)

        x = self.fc2(x)
    #  x = self.softmax(x)

        return x
    
def get_VIT():
    """Loads transformer pretrained on ImageNET and appends MLP model to it

    Returns:
        transformer model
    """
    m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    # adjust to 1 channel data
    m.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
    for param in m.parameters():
        param.requires_grad = False
    mlp = MLP()
    final_model = nn.Sequential(m, mlp)
    return final_model

def adjust_data(batch, device):
    """converts the batch to fit the pretrained transformer

    Args:
        batch: batch of inputs
        device: device to store the batch on

    Returns:
        preprocessed batch
    """
    preprocessing = ViT_B_16_Weights.DEFAULT.transforms()
    preprocessing.mean = preprocessing.mean[0]
    preprocessing.std = preprocessing.std[0]
    return preprocessing(batch).to(device)


class ResidualBlock(nn.Module):
    """
    x
    Conv 3x3 [out_channels]
    Relu
    Conv 3x3 [out_channels]
    += x
    Relu
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=1)
        
        self.skip_connection = nn.Identity()

        # Change number of channels if needed
        self.skip_connection = nn.Identity()
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        residual = self.skip_connection(residual)

        out += residual
        out = self.relu(out)

        return out
    

class ResCNN(nn.module):
    """
    """
    def __init__(self):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1)
        self.resblocks1 = self._get_resblocks(3, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1)
        self.resblock2 = self._get_resblocks(3, 256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1)
        self.resblocks3 = self._get_resblocks(3, 512)

        self.relu = nn.ReLU()

    def _get_resblocks(self, n_blocks, channels):
        """
        Returns Sequential model made of stacked residual blocks
        """

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(channels, channels))
        return nn.Sequential(*blocks)

    def forward(self, x):

        x = self.conv1(x)
        #! Batc-norm
        x = self.relu(x)
        x = self.resblocks1(x)
        x = self.conv2(x)
        #! Batc-norm
        x = self.relu(x)
        x = self.resblock2(x)
        x = self.conv3(x)
        #! Batc-norm
        x = self.relu(x)
        x = self.resblocks3(x)

        x = torch.mean(x, dim=1)