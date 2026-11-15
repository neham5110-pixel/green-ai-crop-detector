# 🌿 Green AI Crop Doctor

Potato leaf disease classification (**Healthy / Early blight / Late blight**) with a small, energy-efficient model (MobileNetV3-Small, exported to TensorFlow Lite), plus a Streamlit demo that turns a diagnosis into a spray plan and a weather safety check.

*Final project, Computer Vision, AI Bootcamp Cohort 18, atomcamp. Author: Neha Muhammad. Instructor: Khawir Mehmood.*

**Live demo:** _add your Streamlit link here_

## 1. The problem and what we found

The first version reached about 98% accuracy on its own test split, yet healthy leaves in real use were labelled as blight. The cause was the data, not the code:

- PlantVillage has only **152 healthy potato images** (1,000 early blight, 1,000 late blight), and only 23 healthy images in the test split.
- PlantVillage photos are all taken on a plain background in even light.
- The model used by the app (full INT8) lost 3.4 points of accuracy and was never checked on the healthy class.

## 2. What was done to fix it

| Change | Reason |
|---|---|
| Added the **PLD potato dataset (Central Punjab, Pakistan)** with about 1,020 healthy leaves; exact duplicates removed | Many more healthy examples, more varied photos |
| Split stratified by class and by source; results reported per source | Shows the model works on both datasets |
| Stronger augmentation (rotation, zoom, shift, brightness, contrast, saturation) | Real photos vary more than lab photos |
| Full per-class report for every model; app model chosen automatically from healthy recall and accuracy | The old INT8 model was never checked on healthy leaves |
| Balanced INT8 calibration; added dynamic-range model | Less accuracy loss when shrinking the model |
| Centre-crop to square, smooth resize (training and app) | Phone photos were being squashed |
| Longer, stronger fine-tuning | The old setting was still improving when it stopped |
| Energy measured per phase, compared fairly with ResNet50 | The old comparison was not like-for-like |
| Streamlit app separated from the training notebook | Accuracy is measured on its own |

## 3. Results

Version 1 (PlantVillage only), measured on 323 test images:

| Model | Size (MB) | Accuracy |
|---|---|---|
| Keras FP32 | 7.13 | 98.1% |
| TFLite FP32 | 3.76 | 97.8% |
| TFLite full INT8 | 1.24 | 94.4% |
| ResNet50 baseline (23.6 M params, 94.4 MB) | 94.4 | 98.1% |

Version 2 (PlantVillage + PLD): _paste the table printed by the notebook here (accuracy, healthy recall, healthy called diseased, per source)._

Energy (CodeCarbon estimate, Colab): the v1 MobileNet total run (head + fine-tune) used 0.00295 kg CO2eq against 0.00231 for the ResNet50 head-only run, so in that run the saving is in **model size and running cost**, not training energy.

## 4. Repository layout

```
notebooks/green_ai_crop_doctor.ipynb   training and evaluation (run on Google Colab)
model/                                 deploy_model.tflite, labels.json, results.json (from the notebook)
app/                                   Streamlit demo (app.py, inference.py, agronomy.py, weather.py)
tests/                                 unit tests (python -m pytest tests)
docs/                                  report and slides
```

## 5. How to reproduce

1. Open `notebooks/green_ai_crop_doctor.ipynb` in Google Colab, set Runtime to GPU, run all cells.
2. Download `model_files.zip` at the end and copy `deploy_model.tflite`, `labels.json`, `results.json` into `model/`.

## 6. Run the demo locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app/app.py
```

## 7. Limitations

- Trained on single-leaf photos; whole-plant or very cluttered photos are out of scope.
- Only three classes. Other diseases or nutrient problems will be forced into one of them (the app asks for a retake when confidence is low).
- Chemical names, doses and prices in `app/agronomy.py` are **example values**. Check the product label or a local agriculture officer before spraying.

## 8. Credits

- PlantVillage dataset (Hughes and Salathe, 2015), via Kaggle `arjuntejaswi/plant-village`.
- Potato Leaf Disease Dataset (PLD), Central Punjab, Pakistan (Rashid et al., 2021).
- Weather: Open-Meteo (free, no key).
