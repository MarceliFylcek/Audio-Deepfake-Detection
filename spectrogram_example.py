from transforms.mel_spectrogram import Mel_Spectrogram

mel = Mel_Spectrogram("resources/song.mp3", 16_000, 128, time_milliseconds=20_000, db_amplitude=True)

mel.play()