import torch
import torchaudio
import torchaudio.transforms as transforms
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import sounddevice as sd

print(torchaudio.get_audio_backend())

class Mel_Spectrogram:
    def __init__(self, audio_path, desired_sample_rate=16_000, n_mels=40, backend="torchaudio"):
        """
        :param n_mels: Number of mel filter banks
        """

        # Load the audio waveform
        self.waveform, self.sample_rate = torchaudio.load(audio_path)

        # Resample the waveform to the desired sample rate
        if self.sample_rate != desired_sample_rate:
            self.waveform = torchaudio.transforms.Resample(self.sample_rate, desired_sample_rate)(self.waveform)
            self.sample_rate = desired_sample_rate

        # Convert the waveform to mono
        waveform_mono = self.waveform.mean(dim=0, keepdim=True)

        # Convert the mono waveform to a NumPy array
        self.audio_array = waveform_mono.numpy()

        #Define the number of mel filterbanks
        n_mels = 128

        #Set parameters
        self.transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=n_mels 
        )

        melspectrogram = self.transform(waveform_mono)

        # Convert the mel spectrogram tensor to a NumPy array
        melspectrogram = melspectrogram.numpy()

        #Drop first dim
        self.melspectrogram = np.squeeze(melspectrogram)

    def play(self):
        current_frame = [0]

        # Animated plot
        def update_plot():
            current_time = current_frame[0] / self.sample_rate
            print(f"{current_time}s")
            marker_position = int(current_frame[0] / self.transform.hop_length)
            
            # Clear the previous marker
            if hasattr(self, 'red_marker_line'):
                self.red_marker_line.remove()
            
            # Add a marker to indicate the current position on the mel spectrogram
            self.red_marker_line = plt.axvline(x=marker_position, color='red')
            
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
        blocksize = 2048

        def callback(outdata, frames, time, status):
            if status.output_underflow:
                print('Output underflow!')
            current_frame[0] += frames
            
        # Play audio with spectrogram visualization
        with sd.OutputStream(callback=callback, channels=num_channels,
                            blocksize=blocksize, samplerate=self.sample_rate):
            sd.play(self.waveform.T, self.sample_rate)
            while sd.get_stream().active:
                plt.pause(0.01)
                update_plot()
                if stop[0]:
                   sd.stop()
                   plt.close()

    def display(self):
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(self.melspectrogram, aspect='auto', origin='lower', vmax=120)
        plt.xlabel('Frames')
        plt.ylabel('Mel Bins')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.show()