# phoneme lipsync

## Run the real-time interface

Install the project dependencies and the PySide6 interface dependency:

```powershell
python -m pip install -r requirements-ui.txt
python -m src.main
```

The app uses the trained checkpoint at
`models/best_cnn_lstm_128_dropout_04_cnn_3.pt`, the phoneme-to-viseme
mapping in `src/viseme_mapping.json`, and the default artwork in
`assets/visemes`. The interface shows all nine visemes in a 3×3 grid beside
the live preview. Enable **Enable custom viseme images** to activate the
per-viseme image buttons; selected files are saved in `assets/custom_visemes`
and replace the corresponding default image while custom images are enabled.

For OBS, click **Open OBS preview** and add a **Window Capture** source for
the separate **LipSync OBS Preview** window.
