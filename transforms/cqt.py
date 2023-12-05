import torch
import torchaudio
import torchaudio.transforms as transforms
from librosa import cqt, note_to_hz
from torchaudio.functional import amplitude_to_DB

import matplotlib.pyplot as plt
import sounddevice as sd


class CQT:
    def __init__(
        self,
        audio_path,
        new_sample_rate,
        n_bins,
        time_milliseconds,
        db_amplitude=False,
        bins_per_octave=12*2,
        fmin=note_to_hz('C1')
    ):

        # Load the audio waveform
        self.waveform, self.sample_rate = torchaudio.load(audio_path)

        # Resample the waveform to the desired sample rate
        if self.sample_rate != new_sample_rate:
            self.waveform = torchaudio.transforms.Resample(
                self.sample_rate, new_sample_rate
            )(self.waveform)
            self.sample_rate = new_sample_rate

        # Convert the waveform to mono
        if self.waveform.shape[0] > 1:
            self.waveform = self.waveform.mean(dim=0, keepdim=True)

        # Cut down length
        if time_milliseconds is not None:
            n_samples = int(self.sample_rate * time_milliseconds / 1000.0)
            signal_length = self.waveform.shape[1]
            if signal_length > n_samples:
                self.waveform = self.waveform[:, :n_samples]
            elif signal_length < n_samples:
                n_missing_samples = n_samples - signal_length
                padding = (0, n_missing_samples)
                self.waveform = torch.nn.functional.pad(self.waveform, padding)

        # Get hop length
        self.hop_length = 512

        # Create Spectogram transform
        cqt_transform = cqt(
            y=self.waveform.numpy(),
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin
        )

        # Create the spectogram
        self.cqt = torch.squeeze(torch.abs(torch.tensor(cqt_transform)))

        if db_amplitude:
            self.cqt = amplitude_to_DB(
                self.cqt,
                multiplier=20.0, # 20 for aplitude to power
                amin=1e-4, # Number to clamp x
                db_multiplier=1.0,
                top_db=80.0,
            )

    def get_raw_data(self):
        return self.cqt

    def _update_plot_marker(self, ax, current_frame):
        """Moves red marker on the plot"""

        # Current time stamp
        current_time = current_frame / self.sample_rate
        print(f"{current_time}s")

        # Current marker postion
        marker_position = int(current_frame / self.hop_length)

        # Clear the previous marker
        if hasattr(self, "red_marker_line"):
            self.red_marker_line.remove()

        # Add marker to current position
        self.red_marker_line = ax.axvline(x=marker_position, color="red")

        # Update the plot
        plt.draw()

    def _outputstream_callback(self, outdata, frames, time, status):
        """ """
        if status.output_underflow:
            print("Output underflow!")
        self.current_frame += frames

    def _on_window_close(self, event):
        self.close = True

    def play(self):
        # Frame counter
        self.current_frame = 0

        # For for closing the window
        self.close = False

        fig, ax = self._create_plot()

        # Detect window closing
        fig.canvas.mpl_connect("close_event", self._on_window_close)

        # Audio settings
        num_channels = 1
        blocksize = 2048

        # Play audio
        with sd.OutputStream(
            callback=self._outputstream_callback,
            channels=num_channels,
            blocksize=blocksize,
            samplerate=self.sample_rate,
        ):
            sd.play(self.waveform.T, self.sample_rate)
            while sd.get_stream().active:
                plt.pause(0.01)
                self._update_plot_marker(ax, self.current_frame)
                if self.close:
                    sd.stop()
                    plt.close()

    def _create_plot(self):
        """Creates a Constant-Q transform plot"""

        # Create figure and ax with a specified size
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot spectogram
        cqt_plot = ax.imshow(
            self.cqt, aspect="auto", origin="lower", cmap="magma"
        )

        # Set label names and title
        ax.set_xlabel("Time bin")
        ax.set_ylabel("Freq bin")
        ax.set_title("Constant-Q transform")

        # Set fig type as colorbar
        fig.colorbar(cqt_plot, format="%+2.0f", cmap="magma")

        return fig, ax

    def display(self):
        """Creates and displays Constant-Q transform plot"""

        # Create the plot
        self._create_plot()

        # Display
        plt.show()

