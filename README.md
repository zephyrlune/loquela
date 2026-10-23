# loquela

Loquela is a desktop app that listens to live audio, predicts phonemes, and maps them to visemes. It displays the current viseme and a grid of viseme images, with a separate preview window for OBS.

## Run

Install the dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m src.main
```

The app uses the model in `models/` and the phoneme-to-viseme mapping in `src/viseme_mapping.json`. A working microphone is required.

To use the preview in OBS, click **Open OBS preview** and add a **Window Capture** source for the **loquela OBS Preview** window.

You can enable custom viseme images in the app and select replacements for the defaults. Custom images are saved in `assets/custom_visemes`.
