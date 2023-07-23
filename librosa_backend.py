import librosa
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
torchaudio.set_audio_backend("soundfile")
import sounddevice as sd

print(torchaudio.get_audio_backend())

class Mel_Spectrogram:
    def __init__(self, audio_path, desired_sample_rate=16_000, n_mels=40):
        """
        :param n_mels: Number of mel filter banks
        """

        # Load the audio waveform
        self.waveform, self.sample_rate = librosa.load(audio_path)

        # Resample the waveform to the desired sample rate
        if self.sample_rate != desired_sample_rate:
            self.waveform = librosa.resample(self.waveform, orig_sr=self.sample_rate, target_sr=desired_sample_rate)
            self.sample_rate = desired_sample_rate

        #Define the number of mel filterbanks
        n_mels = 128

        self.melspectrogram = librosa.feature.melspectrogram(y=self.waveform,
                                                        sr=self.sample_rate,
                                                        n_mels=n_mels)

        self.melspectrogram = librosa.amplitude_to_db(self.melspectrogram, ref=np.max)


    def play(self):       
        time_passed = [0]

        # Animated plot
        def update_plot():
            position_in_frames = int((time_passed[0]*self.sample_rate)/self.transform.hop_length)

            if hasattr(self, 'red_marker_line'):
                self.red_marker_line.remove()

            # Add a marker to indicate the current position on the mel spectrogram
            self.red_marker_line = plt.axvline(x=position_in_frames, color='red')
            
            # Update the plot
            plt.draw()


        # Initialize the plot
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(self.melspectrogram, aspect='auto', origin='lower', vmax=120)
        plt.xlabel('Frames')
        plt.ylabel('Mel Bins')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        stop = [False]
        def onclose(event):
            stop[0] = True

        fig.canvas.mpl_connect('close_event', onclose)

        # Audio settings
        num_channels = 1

        def callback(outdata, frames, time, status):
            if status.output_underflow:
                print('Output underflow!')
            time_passed[0] = time.inputBufferAdcTime
            
        # Play audio with spectrogram visualization
        with sd.OutputStream(callback=callback, 
                             channels=num_channels, blocksize=2048, device=0):
            sd.play(self.waveform.T, self.sample_rate)
            while sd.get_stream().active:
                plt.pause(0.01)
                update_plot()
                if stop[0]:
                   sd.stop()
                   plt.close()

    def display(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(self.melspectrogram,
                                        x_axis="time",
                                        y_axis="log",
                                        ax=ax)
        ax.set_title("MelSpectrogram")
        fig.colorbar(img, ax=ax, format=f'%0.2f')
        plt.show()
            