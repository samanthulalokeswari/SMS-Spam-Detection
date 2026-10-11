
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="SMS Shield | Spam Detection",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


st.title("🛡️ SMS Shield")
st.subheader("AI-Powered SMS Spam Detection")
st.write(
    "Enter a text message and use a trained machine learning "
    "model to classify it as Spam or Not Spam."
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model files are missing. Run python train_model.py first.")
    st.stop()

model = load_model()
metrics = load_metrics()

st.divider()

st.subheader("📩 Test a message")

message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=130
)

if st.button("🔍 Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message.strip()])[0]
        probabilities = model.predict_proba([message.strip()])[0]
        classes = list(model.named_steps["classifier"].classes_)
        spam_probability = float(
            probabilities[classes.index("spam")]
        )

        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
            st.write(
                "This message resembles spam messages learned "
                "from the training dataset."
            )
        else:
            st.success("✅ Prediction: NOT SPAM")
            st.write(
                "The model classified this message as legitimate."
            )

        st.metric(
            "Model-estimated spam score",
            f"{spam_probability * 100:.2f}%"
        )
        st.caption(
            "This score is not necessarily a calibrated probability "
            "or a guarantee. Verify important messages independently."
        )

st.divider()
st.subheader("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
c2.metric(
    "Spam Precision",
    f"{metrics['spam_precision'] * 100:.2f}%"
)
c3.metric(
    "Spam Recall",
    f"{metrics['spam_recall'] * 100:.2f}%"
)
c4.metric("Spam F1-score", f"{metrics['spam_f1'] * 100:.2f}%")

st.caption(
    "Metrics are calculated on the held-out test split, "
    "not on the message currently entered above."
)

st.divider()
st.subheader("📈 Confusion Matrix")

cm = metrics["confusion_matrix"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"],
    ax=ax
)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_title("Test Set Results")
st.pyplot(fig)
plt.close(fig)

st.divider()
st.subheader("🧪 Try Example Messages")

examples = {
    "Normal message": "Hey, are we meeting at college tomorrow?",
    "Potential spam": (
        "Congratulations! You have won a free prize. "
        "Call now to claim your reward!"
    )
}

for label, example in examples.items():
    with st.expander(label):
        st.write(example)
        if st.button("Analyze this example", key=label):
            result = model.predict([example])[0]
            if result == "spam":
                st.error("Prediction: SPAM")
            else:
                st.success("Prediction: NOT SPAM")

st.caption(
    "Educational project | Python • TF-IDF • Naive Bayes • Streamlit"
)
