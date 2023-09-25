from Mel_Spectrogram import Mel_Spectrogram

spectrogram = Mel_Spectrogram("resources/song.mp3", new_sample_rate=16_000, n_mels=64)
# spectrogram.display()
spectrogram.play()




