from Mel_Spectrogram import Mel_Spectrogram

spectrogram = Mel_Spectrogram("resources/LA_E_1000048.flac", desired_sample_rate=16_000, n_mels=64)
spectrogram.play()




