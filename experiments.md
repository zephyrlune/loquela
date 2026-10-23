## Baseline Model — CNN + GRU

### Objective

Establish a baseline real-time phoneme classification model before experimenting with alternative architectures.

### Architecture

```text
80-bin Log-Mel Spectrogram
          ↓
Conv2D — 32 channels
          ↓
Conv2D — 64 channels
          ↓
GRU — 128 hidden units
          ↓
Linear
          ↓
41 phoneme classes
```

The CNN layers extract local acoustic features from the Log-Mel spectrogram, while the unidirectional GRU models temporal relationships between consecutive acoustic frames. The recurrent layer is causal and therefore does not require future audio context, which is important for the real-time application.

### Input Representation

* Sample rate: 16 kHz
* Frame length: 25 ms
* Frame hop: 20 ms
* Mel bins: 80
* FFT size: 400
* Training sequence length: 1 second
* Model input shape: `(80, 49)`
* Output: 49 frame-level predictions per 1-second sequence
* Number of phoneme classes: 41

### Training

* Optimizer: Adam
* Learning rate: `1e-3`
* Batch size: 32
* Loss: Cross-Entropy
* Padding labels: `-100` with `ignore_index=-100`
* Training duration: 10 epochs

### Results

The best observed checkpoint was epoch 9:

* **Test loss:** `0.9114`
* **Test frame accuracy:** `71.89%`

The model was evaluated on speakers that were excluded from training.

### Interpretation

The baseline demonstrated that the CNN-GRU architecture can learn meaningful phoneme representations from the Log-Mel input and achieve approximately 72% frame-level accuracy on unseen speakers.

Several phonemes remained difficult to distinguish, particularly acoustically similar classes. This provides a useful baseline for evaluating subsequent architectural changes.

### Experimental Limitation

At this stage, the dataset was divided into training and test speakers, but a separate validation set was not used. Consequently, the test set was used during experimentation to identify the best-performing epoch.

This means the reported test performance should be treated as a **baseline result rather than a final unbiased evaluation**.

Future experiments will introduce a speaker-level validation split and use validation performance for model selection, while reserving the test speakers for final evaluation.


## Experiment 2 — CNN + LSTM

### Objective

Evaluate whether replacing the recurrent GRU layer with a unidirectional LSTM improves frame-level phoneme classification while maintaining causal real-time inference.

### Architecture

```text
80-bin Log-Mel Spectrogram
        ↓
Conv2D 32
        ↓
Conv2D 64
        ↓
LSTM 128
        ↓
Linear
        ↓
41 phoneme classes
```

The CNN component was kept identical to the previous CNN+GRU experiment. The recurrent layer was changed from GRU to LSTM to isolate the effect of the recurrent architecture.

### Input

* Sample rate: 16 kHz
* Frame length: 25 ms
* Frame hop: 20 ms
* Mel bins: 80
* FFT size: 400
* Chunk duration: 1 second
* Input shape: `(80, 49)`
* Output: 49 frame-level predictions
* Classes: 41

### Dataset Split

The dataset was split by speaker to prevent speakers from appearing across training and evaluation sets.

* Training: approximately 80% of speakers
* Validation: approximately 10% of the training/validation speakers
* Test: 20% of speakers

The validation set was used for model selection. The test set was not used during this experiment.

### Training

* Optimizer: Adam
* Learning rate: 1e-3
* Batch size: 32
* Loss: CrossEntropyLoss
* Ignored label: `-100`
* Epochs: 10
* Recurrent hidden size: 128

### Results

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| ----: | ---------: | --------------: | ------------------: |
|     1 |     0.8480 |          1.1673 |              65.31% |
|     2 |     0.7863 |          1.0559 |              68.91% |
|     3 |     0.7517 |      **0.9719** |          **70.86%** |
|     4 |     0.7301 |          1.0091 |              70.04% |
|     5 |     0.7129 |          0.9963 |              70.38% |
|     6 |     0.6999 |          0.9943 |              70.48% |
|     7 |     0.6902 |          0.9830 |              70.84% |
|     8 |     0.6804 |          0.9996 |              70.50% |
|     9 |     0.6734 |          0.9825 |              70.79% |
|    10 |     0.6673 |      **0.9740** |          **71.05%** |

### Interpretation

The CNN+LSTM model successfully learned the phoneme classification task and reached 71.05% validation frame accuracy by epoch 10.

Validation performance improved rapidly during the first three epochs and then largely plateaued around 70–71%. Training loss continued decreasing throughout the experiment, indicating that the model continued fitting the training data without producing a corresponding improvement in validation performance.

This suggests that the current model may have reached the useful capacity of the 128-unit LSTM configuration and may also be experiencing some degree of overfitting.

The LSTM was selected as the next architecture because it is unidirectional and therefore suitable for causal real-time inference. Unlike a bidirectional recurrent layer, it does not require future audio context.

### Comparison With Previous Baseline

The previous CNN+GRU experiment achieved 71.89% test accuracy at epoch 9. However, that experiment selected the model using the test set because a separate validation split had not yet been implemented.

Therefore, the 71.89% result should be treated as a historical baseline rather than a final unbiased benchmark.

The CNN+LSTM experiment uses the new speaker-level train/validation/test structure and therefore provides a cleaner experimental framework.

### Next Experiment

Test whether increasing the LSTM hidden size from 128 to 192 improves validation performance.

Only the recurrent hidden size will be changed so that the effect of model capacity can be evaluated independently.

## Experiment 3 — CNN + LSTM 192

### Objective

Evaluate whether increasing the LSTM hidden size from 128 to 192 improves frame-level phoneme classification.

The CNN architecture, input representation, optimizer, learning rate, batch size, and dataset split were kept unchanged from the CNN+LSTM-128 experiment.

### Architecture

```text
80-bin Log-Mel Spectrogram
        ↓
Conv2D 32
        ↓
Conv2D 64
        ↓
LSTM 192
        ↓
Linear
        ↓
41 phoneme classes
```

### Training

* Optimizer: Adam
* Learning rate: 1e-3
* Batch size: 32
* Loss: CrossEntropyLoss
* Ignored label: `-100`
* Epochs: 10
* LSTM hidden size: 192
* Unidirectional LSTM for causal real-time inference

### Results

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| ----: | ---------: | --------------: | ------------------: |
|     1 |     1.0009 |          1.1249 |              66.53% |
|     2 |     0.7896 |          1.1019 |              67.43% |
|     3 |     0.7324 |          1.0006 |              70.12% |
|     4 |     0.6997 |          1.0526 |              69.22% |
|     5 |     0.6772 |          1.0329 |              69.78% |
|     6 |     0.6581 |          0.9759 |              71.11% |
|     7 |     0.6449 |          0.9499 |              71.92% |
|     8 |     0.6326 |          0.9409 |              72.28% |
|     9 |     0.6238 |      **0.9266** |          **72.44%** |
|    10 |     0.6138 |          0.9537 |              71.83% |

### Best Checkpoint

The best validation performance occurred at epoch 9:

* Validation loss: **0.9266**
* Validation accuracy: **72.44%**


### Interpretation

Increasing the LSTM hidden size from 128 to 192 improved validation accuracy from 71.05% to 72.44%, an improvement of approximately 1.4 percentage points.

Validation performance continued improving through epoch 9 before declining at epoch 10. This suggests that the larger model has additional useful capacity but may begin to overfit after reaching its best validation performance.

The result also provides evidence that the recurrent component is an important factor in the current architecture.

### Comparison With LSTM 128

| Model          | Hidden Size | Best Validation Accuracy |
| -------------- | ----------: | -----------------------: |
| CNN + LSTM     |         128 |                   71.05% |
| **CNN + LSTM** |     **192** |               **72.44%** |

### Next Experiment

Test a CNN+GRU model with a hidden size of 192.

This will determine whether the improvement from 128 to 192 is specific to the LSTM architecture or primarily caused by increasing recurrent model capacity.

## Experiment 4 — CNN + GRU 192

### Objective

Evaluate whether increasing the GRU hidden size from 128 to 192 produces a comparable improvement to the LSTM architecture.

The CNN architecture and training configuration were kept unchanged.

### Architecture

```text
80-bin Log-Mel Spectrogram
        ↓
Conv2D 32
        ↓
Conv2D 64
        ↓
GRU 192
        ↓
Linear
        ↓
41 phoneme classes
```

### Training

* Optimizer: Adam
* Learning rate: 1e-3
* Batch size: 32
* Loss: CrossEntropyLoss
* Ignored label: `-100`
* Epochs: 10
* GRU hidden size: 192
* Unidirectional GRU for causal real-time inference

### Results

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| ----: | ---------: | --------------: | ------------------: |
|     1 |     0.9847 |          1.1467 |              65.88% |
|     2 |     0.8059 |          1.0729 |              68.18% |
|     3 |     0.7574 |          1.0226 |              69.47% |
|     4 |     0.7294 |          1.0247 |              69.68% |
|     5 |     0.7116 |          1.0313 |              69.19% |
|     6 |     0.6965 |          0.9824 |              70.65% |
|     7 |     0.6864 |      **0.9457** |          **71.77%** |
|     8 |     0.6756 |          1.0008 |              70.26% |
|     9 |     0.6690 |          1.0058 |              70.25% |
|    10 |     0.6610 |          0.9939 |              70.66% |

### Best Checkpoint

The best validation performance occurred at epoch 7:

* Validation loss: **0.9457**
* Validation accuracy: **71.77%**

### Interpretation

Increasing the GRU hidden size to 192 produced a stronger result than the previous 128-unit LSTM experiment, but it did not match the 192-unit LSTM.

The GRU reached a best validation accuracy of 71.77%, compared with 72.44% for the 192-unit LSTM.

This suggests that, for the current CNN architecture and dataset, the LSTM provides a modest advantage over the GRU when both use the same 192-unit hidden size.

The GRU also reached its best validation performance earlier, at epoch 7, followed by a decline in validation performance despite continued improvement in training loss.

### Model Comparison

| Model          | Hidden Size | Best Validation Accuracy |
| -------------- | ----------: | -----------------------: |
| CNN + LSTM     |         128 |                   71.05% |
| **CNN + LSTM** |     **192** |               **72.44%** |
| CNN + GRU      |         192 |                   71.77% |

The current best model is **CNN + LSTM 192**.

### Next Experiment

Test dropout regularization on the CNN + LSTM 192 architecture.

The initial experiment will use dropout of 0.2 between the LSTM and final classifier while keeping the remaining architecture and training configuration unchanged.

The goal is to determine whether regularization can improve validation performance and reduce the observed training/validation gap.

## Experiment 5 — CNN + LSTM 192 + Dropout 0.2

### Objective

Evaluate whether dropout regularization improves the generalization of the CNN+LSTM model.

The previous experiment showed that increasing the LSTM hidden size from 128 to 192 improved validation performance, but the training loss continued decreasing while validation performance began to plateau. Dropout was therefore introduced as a controlled regularization experiment.

### Architecture

```text
80-bin Log-Mel Spectrogram
        ↓
Conv2D 32
        ↓
Conv2D 64
        ↓
LSTM 192
        ↓
Dropout 0.2
        ↓
Linear
        ↓
41 phoneme classes
```

Only one architectural change was made relative to CNN+LSTM-192: a dropout layer with probability 0.2 was added between the LSTM and classifier.

### Training

* Optimizer: Adam
* Learning rate: 1e-3
* Batch size: 32
* Loss: CrossEntropyLoss
* Ignored label: `-100`
* Epochs: 10
* LSTM hidden size: 192
* Dropout: 0.2
* Unidirectional LSTM for causal real-time inference

### Results

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| ----: | ---------: | --------------: | ------------------: |
|     1 |     1.0465 |          1.1457 |              66.32% |
|     2 |     0.8328 |          1.0475 |              68.93% |
|     3 |     0.7774 |          0.9664 |              71.10% |
|     4 |     0.7447 |          0.9726 |              70.92% |
|     5 |     0.7223 |          0.9965 |              70.51% |
|     6 |     0.7048 |          0.9637 |              71.41% |
|     7 |     0.6928 |          0.9635 |              71.52% |
|     8 |     0.6814 |          0.9633 |              71.79% |
|     9 |     0.6719 |          0.9464 |              71.82% |
|    10 |     0.6625 |      **0.9380** |          **72.64%** |

### Best Checkpoint

The best validation performance occurred at epoch 10:

* Validation loss: **0.9380**
* Validation accuracy: **72.64%**


### Interpretation

Adding dropout with probability 0.2 produced a small improvement over the non-regularized CNN+LSTM-192 model.

| Model                            | Best Validation Accuracy |
| -------------------------------- | -----------------------: |
| CNN + LSTM 128                   |                   71.05% |
| CNN + LSTM 192                   |                   72.44% |
| **CNN + LSTM 192 + Dropout 0.2** |               **72.64%** |

The improvement from 72.44% to 72.64% is modest, but validation performance continued improving through epoch 10. Validation loss also reached its lowest value at the final epoch, indicating that the model had not clearly converged by the end of the experiment.

The dropout model also reduced the training/validation gap compared with the non-dropout LSTM-192 model, providing some evidence that regularization is beneficial.

### Next Experiment

Extend the architectural search beyond recurrent capacity and dropout regularization.

The next experiment will evaluate whether a deeper CNN acoustic feature extractor improves phoneme classification. The LSTM hidden size will be reduced to 128 to control computational cost, while a third convolutional layer with 128 channels will be added.

The experiment will also use a higher dropout rate of 0.4 to regularize the larger CNN.

## Experiment 6 — Deeper CNN + LSTM 128 + Dropout 0.4

### Objective

Evaluate whether increasing the depth of the CNN acoustic feature extractor improves phoneme classification performance.

The previous experiment showed that CNN+LSTM-192 with dropout 0.2 achieved 72.64% validation accuracy. Rather than further increasing recurrent capacity, this experiment added a third convolutional layer to give the model greater capacity to learn local acoustic and phonetic patterns.

The LSTM hidden size was reduced from 192 to 128 to limit computational cost, while dropout was increased from 0.2 to 0.4 to provide stronger regularization for the larger CNN.

### Architecture

```text
80-bin Log-Mel Spectrogram
        ↓
Conv2D 32
        ↓
Conv2D 64
        ↓
Conv2D 128
        ↓
LSTM 128
        ↓
Dropout 0.4
        ↓
Linear
        ↓
41 phoneme classes
```

The main architectural change was the addition of a third convolutional block with 128 channels.

The existing two frequency-axis pooling operations were retained, while no additional temporal pooling was introduced so that the model continued to produce predictions for all 49 time frames.

### Training

* Optimizer: Adam
* Learning rate: 1e-3
* Batch size: 32
* Loss: CrossEntropyLoss
* Ignored label: `-100`
* Epochs: 15
* LSTM hidden size: 128
* Dropout: 0.4
* Unidirectional LSTM for causal real-time inference

### Results

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| ----: | ---------: | --------------: | ------------------: |
|     1 |     1.0599 |          1.1050 |              67.34% |
|     2 |     0.8366 |          1.0557 |              68.75% |
|     3 |     0.7794 |          0.9457 |              71.75% |
|     4 |     0.7462 |          0.9314 |              72.31% |
|     5 |     0.7224 |          0.8859 |              73.65% |
|     6 |     0.7051 |          0.9232 |              72.78% |
|     7 |     0.6914 |          0.8728 |              73.96% |
|     8 |     0.6779 |          1.0013 |              71.00% |
|     9 |     0.6680 |          0.8942 |              73.57% |
|    10 |     0.6591 |          0.8749 |              74.02% |
|    11 |     0.6512 |          0.8976 |              74.00% |
|    12 |     0.6429 |          0.9071 |              74.02% |
|    13 |     0.6367 |          0.9337 |              73.26% |
|    14 |     0.6310 |          0.9477 |              72.72% |
|    15 |     0.6245 |          0.9080 |              73.83% |

### Best Checkpoint

The best validation performance occurred at epoch 10:

* Validation loss: **0.8749**
* Validation accuracy: **74.02%**


### Interpretation

Adding a third convolutional layer produced a meaningful improvement over the previous best model.

| Model                                   | Best Validation Accuracy |
| --------------------------------------- | -----------------------: |
| CNN + LSTM 128                          |                   71.05% |
| CNN + LSTM 192                          |                   72.44% |
| CNN + LSTM 192 + Dropout 0.2            |                   72.64% |
| **Deeper CNN + LSTM 128 + Dropout 0.4** |               **74.02%** |

The new model improved validation accuracy from 72.64% to 74.02%, an absolute improvement of **1.38 percentage points**.

The training loss continued decreasing throughout the experiment, while validation performance peaked around epochs 10–12 and became less stable afterward. This indicates that the model has sufficient capacity to fit the training data and shows signs of mild overfitting rather than underfitting.

The improvement is therefore unlikely to come simply from training for more epochs. The additional CNN capacity appears to provide useful acoustic feature extraction, while the larger dropout rate provides regularization for the deeper network.

### Next Experiment

The main architecture components have now been explored sufficiently that further small changes to LSTM size or dropout are unlikely to provide a large improvement.

The next experiment will therefore change the temporal context of the input rather than continuing to tune the existing architecture.

The current model uses approximately 1 second of audio context. A longer context window, such as approximately 1.5 seconds, will be evaluated while retaining the current deeper CNN + LSTM-128 + dropout-0.4 architecture.

This will test whether additional surrounding acoustic context improves recognition of phonemes whose identity depends on neighboring sounds and coarticulation.

## Experiment 7 — Extended 1.5 Second Context

### Objective

Evaluate whether increasing the temporal context from approximately 1.0 second to 1.5 seconds improves phoneme classification by providing more surrounding acoustic information.

The architecture was kept unchanged from Experiment 6 to isolate the effect of the longer input context.

### Architecture

```text
80-bin Log-Mel Spectrogram
        ↓
Conv2D 32
        ↓
Conv2D 64
        ↓
Conv2D 128
        ↓
LSTM 128
        ↓
Dropout 0.4
        ↓
Linear
        ↓
41 phoneme classes
```

### Training Configuration

* Input: 1.5 seconds of audio context
* Features: 80-bin Log-Mel spectrogram
* Temporal resolution: 74 frames
* Classes: 41
* Batch size: 32
* Optimizer: Adam
* Learning rate: 0.001
* Loss: CrossEntropyLoss with `ignore_index=-100`
* LSTM hidden size: 128
* Dropout: 0.4
* Epochs: 10
* Model selection: lowest validation loss
* Test set: untouched

### Results

| Epoch | Train Loss |   Val Loss | Val Accuracy |
| ----: | ---------: | ---------: | -----------: |
|     1 |     1.0589 |     1.0817 |       68.74% |
|     2 |     0.8278 |     0.9932 |       70.99% |
|     3 |     0.7687 |     0.9352 |       72.58% |
|     4 |     0.7323 |     0.9069 |       73.42% |
|     5 |     0.7084 |     0.9052 |       73.65% |
|     6 |     0.6893 |     0.8998 |       73.66% |
|     7 |     0.6756 |     0.9259 |       73.54% |
|     8 |     0.6610 | **0.8899** |   **74.31%** |
|     9 |     0.6501 |     0.9189 |       73.65% |
|    10 |     0.6421 |     0.8999 |       74.15% |

### Best Result

* Best validation loss: 0.8899 at epoch 8
* Best validation accuracy: 74.31% at epoch 8
* Epoch 10 accuracy: 74.15%

### Interpretation

Increasing the context from approximately 1.0 second to 1.5 seconds produced only a small improvement in validation accuracy.

The best 1.5-second validation accuracy was 74.31%, compared with 74.02% for the previous 1.0-second model. This represents an improvement of only **0.29 percentage points**.

The best validation loss also did not improve: the 1.0-second model achieved 0.8749, compared with 0.8899 for the 1.5-second model.

Therefore, the longer context does not provide strong evidence of a meaningful improvement. It also increases the amount of temporal computation required for real-time inference.

For the current project, the **1.0-second context will remain the preferred practical baseline**.

### Conclusion

The experiment suggests that simply increasing temporal context is not an effective way to substantially improve phoneme classification for this model.

The previous deeper CNN + LSTM-128 architecture with 1.0-second context remains the stronger practical configuration because it achieves comparable validation accuracy with less temporal computation.


### Next Experiment

The next experiment will investigate **audio augmentation** rather than increasing model capacity or temporal context.

The goal is to improve robustness to conditions that differ from the relatively clean LibriSpeech recordings used for training.

Training samples will be augmented with realistic variations such as background noise, gain/volume changes, and mild speaking-rate variation.

The model architecture will remain the deeper CNN + LSTM-128 + dropout-0.4 architecture with approximately 1.0 second of context.


