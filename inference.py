import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torch.nn import Module
from transformers import Dinov2Config

from transforms.mel_spectrogram import Mel_Spectrogram
from models import DinoV2TransformerBasedModel


def save_mosaic(data_list, rows, one_img_size, break_thickness, out_dir, spectrogram=None, imshow=False):
    if spectrogram is not None:
        data_list.append(spectrogram)
    size = one_img_size
    break_size = break_thickness
    cols = len(data_list) // rows if not len(data_list) // rows < len(data_list) else len(
        data_list) // rows + 1

    shape = (rows * (size[0] + break_size), cols * (size[1] + break_size))
    shape = shape + (data_list[0].shape[-1],) if len(data_list[0].shape) > 2 else shape
    output = np.zeros(shape)
    for row in range(rows):
        for col in range(cols):
            if not col + row * cols < len(data_list):
                break
            output[row * (size[0] + break_size):row * (size[0] + break_size) + size[0],
                   col * (size[1] + break_size):col * (size[1] + break_size) + size[1]] = \
                cv2.resize(data_list[col + row * cols], size)

    if imshow:
        cv2.imshow("", output)
        cv2.waitKey(0)
    else:
        cv2.imwrite(out_dir, np.uint8(np.clip(output * 255, 0, 255)))

    if spectrogram is not None:
        del data_list[-1]


def get_attention_map(attentions, patches_shape, patch_size):
    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, *patches_shape)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0),
                                                 scale_factor=patch_size, mode="bilinear")[0].detach().cpu().numpy()

    return attentions


def visualize_all_attentions(attentions, patches_shape, patch_size, out_root, spectrogram=None, imshow=False):
    cmap = plt.get_cmap("magma")
    if spectrogram is not None:
        normalized_spectrogram = Normalize(vmin=spectrogram.min(), vmax=spectrogram.max())(spectrogram)
        spectrogram = cmap(normalized_spectrogram)[::-1, :, :3].astype(np.float32)

    attention_prob_sums = []
    for i, attention in enumerate(attentions):
        attention_map = get_attention_map(attention, patches_shape, patch_size)
        not_norm_attention_probs = []
        attention_probs = []
        for att_map in attention_map:
            not_norm_attention_probs.append(np.copy(att_map))
            att_map = Normalize(vmin=att_map.min(), vmax=att_map.max())(att_map)
            att_map = cv2.cvtColor(cmap(att_map)[::-1, :, :3].astype(np.float32), cv2.COLOR_RGB2BGR)
            attention_probs.append(att_map)
        # save_mosaic(attention_probs, 3, (220, 220), 2, os.path.join(out_root, f"attention_probs_{i}.png"),
        #             spectrogram, imshow=imshow)
        att_prob_sum = sum(not_norm_attention_probs)
        att_prob_sum = Normalize(vmin=att_prob_sum.min(), vmax=att_prob_sum.max())(att_prob_sum)
        att_prob_sum = cv2.cvtColor(cmap(att_prob_sum)[::-1, :, :3].astype(np.float32), cv2.COLOR_RGB2BGR)
        attention_prob_sums.append(att_prob_sum)
    save_mosaic(attention_prob_sums, 3, (220, 220), 2, os.path.join(out_root, f"attention_prob_sums.png"),
                spectrogram, imshow=imshow)


@torch.no_grad()
def test_voice(model: Module, path_to_voice_file: str, time_milliseconds: int, sampling_rate: int, n_mels: int):
    print(f"Testing {path_to_voice_file}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    model.to(device)

    spectrogram = Mel_Spectrogram(path_to_voice_file, sampling_rate, n_mels, time_milliseconds)
    # spectrogram.play()
    spectrogram_data = spectrogram.get_raw_data().unsqueeze(0).unsqueeze(0).to(device)

    class_values, attentions = model(spectrogram_data)
    print(f"Audio is {'real' if torch.argmax(class_values) == 1 else 'deepfake'}")

    visualize_all_attentions([att.detach().cpu() for att in attentions],
                             model.get_patches_shape(spectrogram_data.shape[-2:]), model.patch_size, "", imshow=True,
                             spectrogram=spectrogram_data[0, 0].detach().cpu().numpy())


def saliency(model: Module, path_to_voice_file: str, time_milliseconds: int, sampling_rate: int, n_mels: int):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    spectrogram = Mel_Spectrogram(path_to_voice_file, sampling_rate, n_mels, time_milliseconds)
    spectrogram_data = spectrogram.get_raw_data().unsqueeze(0).unsqueeze(0).to(device)

    # we want to calculate gradient of the highest score w.r.t. input
    # so set requires_grad to True for input
    spectrogram_data.requires_grad = True

    # forward pass to calculate predictions
    class_values = model(spectrogram_data)
    score, indices = torch.max(class_values, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(spectrogram_data.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    cmap = plt.get_cmap("magma")
    normalized_spectrogram = Normalize(vmin=spectrogram_data.min(), vmax=spectrogram_data.max())(spectrogram_data.detach().cpu().numpy()[0, 0])
    spectrogram = cmap(normalized_spectrogram)[::-1, :, :3].astype(np.float32)

    # plot image and its saleincy map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.detach().cpu().numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    print("Loading model...")
    # m = CNNModel(n_filters=25, input_shape=(64, 126))
    # m.load_state_dict(torch.load(r"C:\Code\Audio-Deepfake-Detection\models\elevenlabsFLAC_p4_e20.pth.pth"))
    m = DinoV2TransformerBasedModel(Dinov2Config(num_channels=1, patch_size=4, hidden_size=48), input_shape=(64, 126))
    m.load_state_dict(torch.load(r"C:\Code\Audio-Deepfake-Detection\models\elevenlabsFLAC_p4_e20.pth"))
    saliency(m, r"C:\Code\Audio-Deepfake-Detection\elevenlabs\valid\real\61943-0003.flac", 4_000, 16_000, 64)
    # test_voice(m, r"C:\Code\Audio-Deepfake-Detection\elevenlabs\valid\real\61943-0003.flac", 4_000, 16_000, 64)
