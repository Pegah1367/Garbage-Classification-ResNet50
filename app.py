import streamlit as st
import numpy as np
import json
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================================================
#     FIX FOR Lambda + preprocess_input DESERIALIZATION
# =========================================================
tf.keras.config.enable_unsafe_deserialization()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "resnet50_garbage.keras",
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )
    return model

@st.cache_resource
def load_classes():
    with open("idx_to_class.json", "r") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return idx_to_class


Model = load_model()
idx_to_class = load_classes()
class_labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

# =========================================================
#                    STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Garbage Classification", page_icon="♻️")

st.title("♻️ Garbage Classification Demo")
st.write(
    "Upload an image of garbage (metal, glass, paper, trash, cardboard, or plastic) "
    "and let the AI classify it!"
)

uploaded_file = st.file_uploader(
    "Upload an Image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------- Preprocess ----------------
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- Predict ----------------
    preds = Model.predict(img_array)
    preds = preds[0]  # shape = (6,)

    # Top-3 predictions
    top_k = 3
    top_indices = np.argsort(preds)[::-1][:top_k]

    st.subheader("Prediction")

    for idx in top_indices:
        label = class_labels[idx]
        confidence = preds[idx]
        st.write(f"**{label}** : {confidence*100:.2f}%")

    # ---------------- Probability Table ----------------
    st.write("### All Class Probabilities")
    prob_dict = {class_labels[i]: float(preds[i]) for i in range(len(class_labels))}

    st.table(
        [{"Class": cls, "Confidence": f"{prob_dict[cls]*100:.2f}%"}
         for cls in class_labels]
    )
