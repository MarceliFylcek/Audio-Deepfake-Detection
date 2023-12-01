# Where models are saved and loaded from
MODELS_DIR = "models"
# Training dataset
TRAIN_DIR = "elevenlabs/train"
# Validation dataset
VALID_DIR = "elevenlabs/valid"

# Set Mel Spectrogram parameters
melspectogram_params = {
    'new_sample_rate': 16_000,
    'n_bins': 128,
    'time_milliseconds': 4_000,
    'db_amplitude': False
}

# Set Mel Spectrogram parameters
melspectogram_params_vit16 = {
    'new_sample_rate': 16_000,
    'n_bins': 224,
    'time_milliseconds': 7_136,
    'db_amplitude': True
}

# output shape of spectrogram = [n_fft/2 + 1, (time*sample_rate)/hop_length]
# output shape of melspectrogram = [n_mels, (time*sample_rate)/hop_length]
# output shape of MFCC = [n_mfcc, (time*sample_rate)/hop_length]

# n_bins = n_mels = n_mfcc = (n_fft-1)*2