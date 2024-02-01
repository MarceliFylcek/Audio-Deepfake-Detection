# Audio-Deepfake-Detection

#### Available models
- Dense Network
- CNN
- CNN+LSTM
- Vision Transformer

#### Available audio features
- CQT
- LPC
- Mel Spectrogram
- Spectrogram

### Training

- Training function inside main.py:
```
if __name__ == "__main__":
    train("CNN", TRAIN_DIR_11LABS, [VALID_DIR_11LABS, VALID_DIR_COREJ], Mel_Spectrogram, False, "Mel_Spectrogram")
```

- Run main.py with chosen training parameters:

```
python main.py --name CNN_model --n_epochs 10 --batch_size 128
```

