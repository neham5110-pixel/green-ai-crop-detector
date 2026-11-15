import os

import streamlit as st
from PIL import Image

from agronomy import DISCLAIMER, calculate_prescription, class_to_disease
from inference import load_interpreter, load_labels, predict_tta, preprocess
from weather import check_spray_safety, get_coordinates, get_weather

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "deploy_model.tflite")
LABELS_PATH = os.path.join(ROOT, "model", "labels.json")
CONFIDENCE_THRESHOLD = 0.70
HEALTHY_THRESHOLD = 0.85

st.set_page_config(page_title="Green AI Crop Doctor", page_icon="🌿", layout="centered")


@st.cache_resource
def load_model():
    return load_interpreter(MODEL_PATH), load_labels(LABELS_PATH)


st.title("🌿 Green AI Crop Doctor")
size_mb = os.path.getsize(MODEL_PATH) / 1e6 if os.path.exists(MODEL_PATH) else 0.0
st.caption(f"Lightweight TFLite model ({size_mb:.1f} MB): low energy, runs on a normal CPU.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
    st.error(
        f"Model files not found. Put `{MODEL_PATH}` and `{LABELS_PATH}` "
        "(created by the training notebook) next to app.py."
    )
    st.stop()

interpreter, class_names = load_model()


with st.sidebar:
    st.header("Field details")
    area = st.number_input("Field area", min_value=0.1, value=2.0, step=0.5)
    unit = st.selectbox("Unit", ["acre", "hectare"])
    tank = st.number_input("Sprayer tank size (liters)", min_value=1.0, value=20.0, step=1.0)
    city = st.text_input("Nearest city (for weather)", value="Multan")


st.subheader("1. Upload a leaf photo")
source = st.radio("Photo source", ["Upload file", "Use camera"], horizontal=True)
file = (
    st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
    if source == "Upload file"
    else st.camera_input("Take a photo of one leaf")
)

if file is None:
    st.info("Upload a clear, close-up photo of a single leaf to begin.")
    st.stop()

image = Image.open(file)
st.image(image, caption="Your photo", use_container_width=True)

with st.spinner("Analysing leaf..."):
    probs = predict_tta(interpreter, preprocess(image))

best = int(probs.argmax())
confidence = float(probs[best])
class_name = class_names[best]

st.subheader("2. Diagnosis")
disease = class_to_disease(class_name)
threshold = HEALTHY_THRESHOLD if disease == "Healthy" else CONFIDENCE_THRESHOLD
if confidence < threshold:
    st.warning(
        f"Not sure (best guess {confidence:.0%}). Please retake the photo: "
        "one leaf, good light, leaf filling most of the frame."
    )
    st.bar_chart({n: float(p) for n, p in zip(class_names, probs)})
    st.stop()

st.metric("Result", disease, f"{confidence:.0%} confidence")
st.bar_chart({n: float(p) for n, p in zip(class_names, probs)})


st.subheader("3. Spray plan")
try:
    rx = calculate_prescription(disease, area, unit, tank)
except ValueError as e:
    st.error(str(e))
    st.stop()

if not rx["needs_treatment"]:
    st.success(rx["message"])
else:
    st.error(rx["message"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Chemical", rx["chemical"])
    c2.metric("Total chemical", f"{rx['chemical_grams']} g")
    c3.metric("Estimated cost", f"PKR {rx['cost_pkr']:,.0f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Water needed", f"{rx['water_liters']} L")
    c5.metric("Tanks", rx["total_tanks"])
    c6.metric("Chemical per tank", f"{rx['grams_per_tank']} g")


    st.subheader("4. Is it safe to spray now?")
    geo = get_coordinates(city)
    if not geo["success"]:
        st.warning(f"Could not find weather for '{city}': {geo['error']}")
    else:
        wx = get_weather(geo["latitude"], geo["longitude"])
        if not wx["success"]:
            st.warning(f"Weather service unavailable: {wx['error']}")
        else:
            st.caption(f"Weather for {geo['city']}, {geo['country']}")
            w1, w2, w3 = st.columns(3)
            w1.metric("Wind", f"{wx['wind_kmh']} km/h")
            w2.metric("Rain chance (6h)", f"{wx['rain_prob']}%")
            w3.metric("Temperature", f"{wx['temp_c']} °C")
            verdict = check_spray_safety(wx["wind_kmh"], wx["rain_prob"], wx["temp_c"])
            (st.success if verdict["is_safe"] else st.error)(f"Spraying: {verdict['status']}")
            for note in verdict["warnings"]:
                st.write(f"- {note}")

st.divider()
st.caption(DISCLAIMER)
