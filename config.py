# Where models are saved and loaded from
MODELS_DIR = "models"
# Training dataset
TRAIN_DIR = "elevenlabs/train"
# Validation dataset
VALID_DIR = "elevenlabs/valid"

# Set Mel Spectrogram parameters
melspectogram_params = {
    'new_sample_rate': 16_000,
    'n_mels': 128,
    'time_milliseconds': 4_000,
    'db_amplitude': True
}
