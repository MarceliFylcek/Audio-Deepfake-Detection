# Where models are saved and loaded from
MODELS_DIR = "models"
# Training dataset
TRAIN_DIR = "resources/train"
 # Validation dataset 
VALID_DIR = "resources/valid"

# Set Mel Spectrogram parameters
melspectogram_params = {
    'new_sample_rate': 16_000, 
    'n_mels': 128,
    'time_milliseconds': 4_000, 
    'db_amplitude': True
}

# Set Mel Spectrogram parameters
melspectogram_params_vit16 = {
    'new_sample_rate': 16_000, 
    'n_mels': 224, 
    'time_milliseconds': 7_136, 
    'db_amplitude': True
}
