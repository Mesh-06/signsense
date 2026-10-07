# SignSense
 
Real-time traffic sign recognition with voice alerts, running entirely in the browser.
 
SignSense takes a photo or camera frame of a traffic sign, identifies it among 43 classes from the German Traffic Sign Recognition Benchmark (GTSRB), and announces it aloud. Critical signs such as Stop and Yield get urgent alerts, and speed-limit signs are compared against the speed the user enters. There is no backend: the model is converted to TensorFlow.js and inference happens on the user's device.
 
**Live demo:** `https://mesh-06.github.io/signsense/`
 
---
 
## Contents
 
1. [Motivation](#motivation)
2. [What it does](#what-it-does)
3. [System overview](#system-overview)
4. [Dataset](#dataset)
5. [Model selection](#model-selection)
6. [Training configuration](#training-configuration)
7. [Results](#results)
8. [Browser embedding](#browser-embedding)
9. [Running locally](#running-locally)
10. [Repository layout](#repository-layout)
11. [Limitations](#limitations)
12. [Future work](#future-work)
13. [Acknowledgements and citation](#acknowledgements-and-citation)
---
 
## Motivation
 
Missed or misread road signs are a recurring factor in accidents, particularly for tired or distracted drivers and for new drivers. A system that reads a sign and says it aloud is a small but concrete aid. This project builds that as a self-contained web page that needs no installation, no account and no server.
 
## What it does
 
- Classifies a sign from a live camera, an uploaded photo, or a built-in set of sample images.
- Shows the top three predictions with confidence scores.
- Rejects low-confidence predictions using an adjustable threshold.
- Smooths predictions over time: an alert fires only when the same sign wins in at least 6 of the last 8 frames.
- Speaks alerts through the Web Speech API, with a five-second cooldown per sign.
- Warns when the entered vehicle speed exceeds a detected speed limit.
- Includes an in-app parity test that checks the browser's predictions against the Python predictions recorded during training.
### Alert categories
 
| Category | Signs | Behaviour |
|---|---|---|
| Critical | Stop, Yield, No entry | Urgent voice, interrupts any speech in progress |
| Speed | Speed limits | Reads the limit and compares it with the entered speed |
| Warning | General caution through wild animals crossing | Prefixed with "Caution" |
| Prohibitory | No passing, no vehicles, and similar | Normal voice |
| Mandatory | Turn, keep, roundabout | Normal voice |
| Information | End-of-limit signs, priority road | Log entry only |
 
## System overview
 
Training pipeline:
 
```
GTSRB images -> crop to sign box -> resize to 48x48 -> group-aware split
  -> augmentation (training set only) -> train five architectures
  -> evaluate, robustness tests -> weighted selection matrix
  -> export to TensorFlow.js -> browser app -> parity test
```
 
Runtime pipeline in the browser:
 
```
camera / upload / sample -> centre-square crop at native resolution
  -> resize to 48x48 (bilinear, half-pixel centres)
  -> CNN with rescaling layer inside the model -> softmax over 43 classes
  -> confidence threshold -> temporal smoothing (6 of last 8 frames)
  -> category lookup -> voice alert
```
 
SignSense is a **classifier**, not a detector. It expects the sign to be roughly centred in the frame, which is why the interface provides an aiming reticle. Finding signs in a full street scene would require an additional detection stage.
 
## Dataset
 
The project uses the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html).
 
| Property | Value |
|---|---|
| Training images | 39,209 |
| Test images | 12,630 |
| Classes | 43 |
| Image size | Varies from roughly 15x15 to 250x250 pixels |
| Class balance | Imbalanced, about 10:1 between the largest and smallest class |
 
Training used the Kaggle distribution of the dataset, which provides `Train.csv`, `Test.csv` and image folders with region-of-interest boxes.
 
### Preprocessing decisions
 
| Decision | Reason |
|---|---|
| Crop to the region-of-interest box | Removes background so the network learns the sign itself |
| Resize to 48x48 | Preserves digit detail on speed limits that 32x32 can blur, while staying cheap in the browser |
| Keep RGB | Sign meaning is colour-coded (red for prohibition, blue for mandatory) |
| Keep float32, do not cast to uint8 | Matches what the browser computes |
| Rescaling layer inside the model | The browser passes raw 0-255 pixels, so no separate normalisation code can drift |
 
### Avoiding train/validation leakage
 
Each physical sign in GTSRB was photographed as a track of about 30 consecutive, near-identical frames. A random split places sibling frames on both sides, so validation accuracy is inflated. SignSense instead splits by track using `StratifiedGroupKFold`, with the track identifier taken from the file name. The notebook asserts that no track appears in both sets. The official test set comes from different tracks and is untouched until final evaluation.
 
### Augmentation
 
Rotation (up to 15 degrees), translation (10 percent), zoom (plus or minus 15 percent), contrast, brightness and Gaussian noise are applied to training data only. Horizontal flipping is deliberately excluded, because it would turn "Turn left ahead" into "Turn right ahead" and "Keep left" into "Keep right", corrupting the labels.
 
## Model selection
 
Selection criteria and weights were fixed before any results were seen.
 
| Criterion | Weight | Rationale |
|---|---|---|
| Macro-F1 on the test set | 30% | Counts rare classes equally on an imbalanced dataset |
| Robustness | 15% | Mean accuracy under darkening, brightening, noise, blur, low resolution and occlusion |
| Browser latency | 15% | Alerts need to feel immediate |
| Model size | 15% | Download time on mobile connections |
| Deployability | 15% | Clean TensorFlow.js conversion without custom operations |
| Training cost | 10% | Time constraint |
 
Recall on safety-critical classes (Stop, Yield, No entry, Pedestrians, Children crossing) is used as the tie-breaker, since a missed Stop sign is more costly than a missed "Bumpy road".
 
### Candidates
 
| Model | Purpose in the comparison |
|---|---|
| LeNet-style | Classic small baseline |
| Custom CNN (three convolutional blocks, batch normalisation, global average pooling) | Purpose-built for small 48x48 inputs |
| ResNet-18, trained from scratch | Tests whether depth and residual connections help |
| MobileNetV2, ImageNet transfer learning | Tests pretrained features on small sign crops |
| Spatial-transformer CNN | Custom CNN plus a learned geometric normaliser; a controlled one-change comparison |
 
Each metric is min-max scaled to a 0-10 range, multiplied by its weight and summed. The deployability score is a judgement rather than a measurement; the spatial-transformer model is penalised because its grid-sampling layer is a custom operation that TensorFlow.js does not provide.
 
**Deployed model:** Custom CNN (three convolutional blocks). Full selection matrix: see [Results](#results).
 
## Training configuration
 
All models were trained under one protocol so that differences come from architecture rather than tuning.
 
| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 1e-3 |
| Loss | Sparse categorical cross-entropy |
| Batch size | 128 |
| Maximum epochs | 40 |
| Early stopping | Validation loss, patience 8, restore best weights |
| Learning-rate schedule | ReduceLROnPlateau, factor 0.5, patience 3 |
| Regularisation | Batch normalisation, dropout (0.25 convolutional, 0.4 dense), augmentation |
| Random seed | 42 |
 
MobileNetV2 is trained in two phases. The new classification head is trained first with the backbone frozen (10 epochs), then the whole network is fine-tuned at a learning rate of 1e-4 (up to 25 epochs). This keeps large, noisy gradients from a randomly initialised head away from the pretrained features.
 
## Results
 
All figures below must come from your own runs. Replace each `TBD` with the measured value from `results.csv`, `robustness.csv` and `selection_matrix.csv`.
 
### Test-set comparison
 
| Model | Parameters | Size (MB) | Test accuracy | Macro-F1 | Critical recall | Robustness (mean) | Browser latency (ms) | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| LeNet-style | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Custom CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet-18 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNetV2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Spatial-transformer CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
 
### Robustness (accuracy under perturbation)
 
| Model | Dark | Bright | Noise | Blur | Low-res | Occlusion |
|---|---|---|---|---|---|---|
| LeNet-style | TBD | TBD | TBD | TBD | TBD | TBD |
| Custom CNN | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet-18 | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNetV2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Spatial-transformer CNN | TBD | TBD | TBD | TBD | TBD | TBD |
 
### Selection matrix
 
| Model | Macro-F1 | Robustness | Latency | Size | Deployability | Training | Weighted total |
|---|---|---|---|---|---|---|---|
| LeNet-style | TBD | TBD | TBD | TBD | 10 | TBD | TBD |
| Custom CNN | TBD | TBD | TBD | TBD | 10 | TBD | TBD |
| ResNet-18 | TBD | TBD | TBD | TBD | 10 | TBD | TBD |
| MobileNetV2 | TBD | TBD | TBD | TBD | 10 | TBD | TBD |
| Spatial-transformer CNN | TBD | TBD | TBD | TBD | 2 | TBD | TBD |
 
### Most confused class pairs
 
Add the top confusions from the error-analysis cell and a short explanation of each.
 
### Ablations
 
Add only the ablations you actually ran (for example augmentation on/off, batch normalisation, input size), with the measured effect of each.
 
## Browser embedding
 
The trained Keras model is exported as a TensorFlow SavedModel and converted with `tensorflowjs_converter` to a TensorFlow.js graph model (`model.json` plus binary weight shards) with float16 quantisation. The converted weights total about 1.3 MB, which matches the size expected from the Keras parameter count.
 
The main embedding risk is a mismatch between training and browser preprocessing, which degrades accuracy without raising an error. It is controlled as follows.
 
| Concern | Approach |
|---|---|
| Resize differences | `tf.image.resize` (bilinear, no antialiasing) in Python and `tf.image.resizeBilinear(x, size, false, true)` (half-pixel centres) in JavaScript |
| Normalisation differences | The `Rescaling` layer is part of the exported model, so the browser supplies raw 0-255 values |
| Browser rescaling of camera frames | Frames are drawn to a canvas at native resolution and cropped before resizing |
| Memory growth in the camera loop | Tensors are created inside `tf.tidy` and outputs are disposed |
| Cold-start delay | A warm-up inference runs when the model loads |
 
### Parity test
 
During training, 12 test-set crops were predicted in Python and stored with their predicted class and probability in `samples/expected.json`. The **Run embedding parity test** button on the Samples tab runs the same images through the browser pipeline and checks that the class matches and the probability differs by less than 0.02.
 
Parity result: `TBD / 12 match` (record your observed result here).
 
### Privacy
 
All inference runs on the user's device. Images and camera frames are never uploaded.
 
## Running locally
 
The page must be served over HTTP; opening `index.html` directly from the file system will not load the model.
 
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m http.server 8000
```
 
Open `http://localhost:8000`. The camera API works on `localhost` and on any HTTPS origin, including GitHub Pages.
 
On iOS, voice output starts only after the user taps the voice button, which the app handles.
 
### Using the app
 
1. Wait for the status line to read "Model ready" together with the active backend.
2. Choose a tab: **Camera** (hold a sign image or printout inside the reticle), **Upload** (frame the sign with the zoom and position sliders), or **Samples** (tap a test image).
3. Tap the voice button to enable spoken alerts.
4. Adjust the minimum-confidence slider and, optionally, enter a vehicle speed.
### Retraining
 
The training notebook covers data loading, the grouped split, all five models, evaluation, robustness tests, the selection matrix, export and parity-asset generation. It is designed for a Google Colab GPU runtime. Dataset: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign` on Kaggle, downloaded with `kagglehub`.
 
## Repository layout
 
```
.
├── index.html            Page structure
├── style.css             Styling
├── app.js                Inference, smoothing, alerts, parity test
├── model/
│   ├── model.json        TensorFlow.js graph definition
│   ├── group1-shard*.bin Quantised weights
│   └── info.json         Model name, parameter count, test metrics
├── samples/
│   ├── s0.png ... s11.png
│   └── expected.json     Python predictions used by the parity test
└── README.md
```
 
## Limitations
 
- **German signs only.** GTSRB covers German signs. Symbols, layouts and conventions differ elsewhere, so accuracy on other countries' roads, including India, is not expected to hold without fine-tuning on local data.
- **Classifier, not detector.** The sign must be roughly centred and cropped. Full-scene detection needs a separate detector stage.
- **Single-frame, smoothed predictions.** Requiring 6 of 8 frames suppresses flicker but can miss a sign that passes quickly.
- **Image conditions.** Poor lighting, motion blur and partial occlusion reduce accuracy; the robustness table quantifies this.
- **No "no sign" class in the deployed model.** The deployed model has 43 outputs, so any input receives a sign label unless it falls below the confidence threshold. A 44-class variant trained with negative examples is a possible extension, and `BG` in `app.js` is set up for it; it is inactive for the 43-class model.
- **Not a safety system.** This is an assistive demonstration and has not been validated or certified for use while driving.
## Future work
 
- A detector-plus-classifier pipeline for full-scene recognition.
- Fine-tuning on an Indian traffic-sign dataset, subject to its licence.
- Deploying the 44-class model with a measured false-alarm comparison against threshold-only rejection.
- An on-device TensorFlow Lite application.
- Reading supplementary plates with OCR.
## Acknowledgements and citation
 
Dataset: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition." *Neural Networks*, 32, 2012.
 
Built with TensorFlow, Keras, scikit-learn, TensorFlow.js and the Web Speech API. Trained on Google Colab and hosted on GitHub Pages.
 
## Licence
 
Add a licence of your choice (for example MIT) as a `LICENSE` file, and note that the GTSRB dataset has its own terms of use.
 
