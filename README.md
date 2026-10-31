# loquela

https://github.com/user-attachments/assets/c6b91acd-8f3d-481f-afc4-67b9d11d6ca3

Loquela listens to a selected microphone, predicts phonemes from live audio, and displays a corresponding viseme image. The interface shows the current prediction, microphone level, and available visemes.

## How it works

Audio is captured as mono at 16 kHz. The app keeps a rolling one-second audio window and converts it into an 80-bin log-mel spectrogram with 49 time frames. A CNN-LSTM model predicts a phoneme for the latest frame. The mapping in `src/viseme_mapping.json` maps phonemes to viseme names, and the app displays the matching image from `assets/visemes`.

The model in `models/best_cnn_lstm_128_dropout_04_cnn_3.pt` has three convolutional layers with 32, 64, and 128 channels. The CNN extracts features from the spectrogram, a 128-unit LSTM processes them over time, followed by dropout and a linear classifier for 41 phoneme classes. Inference uses CUDA when PyTorch reports it is available, otherwise it uses the CPU.

## Microphone and noise gate

Choose an input device from the microphone combo box. The noise-gate slider mutes audio below its threshold with the default being -45 dBFS. Use **Start listening** and **Stop** to control mic stream.

## Custom visemes

Choose **Enable custom viseme images** to use custom visemes. Choose an image for a viseme from PNG, JPG, JPEG, WEBP, or BMP files. Custom images take precedence over the defaults while custom mode is enabled.

During development, custom images are stored in `assets/custom_visemes`. In the packaged Windows app, they are stored in `%APPDATA%\Loquela\custom_visemes` so the app can write them without changing its installation files.

## Add as OBS source

Click **Open OBS preview** to open the separate preview window. In OBS, add a **Window Capture** source for **loquela OBS Preview** and set **Capture Method** to be **Windows 10** in the source settings. The title strip can be cropped out in the capture

## Run from source

Install the dependencies and start the app from the project root:

```powershell
python -m pip install -r requirements.txt
python -m src.main
```

The app requires a working microphone and the model checkpoint in `models/`.

## Build a Windows application folder

Install the app and build dependencies, then run PyInstaller from the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
python -m PyInstaller --noconfirm --clean .\packaging\loquela.spec
```

The executable is `dist/Loquela/Loquela.exe`. Keep the whole `dist/Loquela` folder together when moving or distributing the app.
