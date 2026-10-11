
from pathlib import Path
import json

import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="SMS Shield | AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PREMIUM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #080D19;
    --panel: #10192B;
    --panel2: #131F34;
    --border: rgba(151, 174, 218, .16);
    --blue: #60A5FA;
    --cyan: #67E8F9;
    --white: #F1F5F9;
    --muted: #91A2BD;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #142746 0%, transparent 36%),
        radial-gradient(ellipse at 95% 15%, #10273A 0%, transparent 30%),
        var(--bg);
    color: var(--white);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--white) !important;
}

p, label, li {
    color: #B8C5D9;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 5px 0 28px 0;
}

.brand-icon {
    width: 48px;
    height: 48px;
    border-radius: 15px;
    background: linear-gradient(135deg, #2563EB, #06B6D4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    box-shadow: 0 8px 30px rgba(37, 99, 235, .28);
}

.brand-name {
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -.8px;
}

.brand-sub {
    color: #8194B3;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 3px;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    border: 1px solid var(--border);
    background: linear-gradient(120deg,
        rgba(22, 40, 69, .96),
        rgba(13, 28, 48, .88));
    position: relative;
    overflow: hidden;
    margin-bottom: 25px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -65px;
    top: -90px;
    border-radius: 50%;
    background: rgba(56, 189, 248, .09);
    border: 1px solid rgba(103, 232, 249, .12);
}

.eyebrow {
    display: inline-block;
    padding: 7px 12px;
    border: 1px solid rgba(103, 232, 249, .23);
    border-radius: 50px;
    color: #8DEBFA;
    background: rgba(8, 145, 178, .10);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(30px, 4vw, 45px);
    line-height: 1.15;
    letter-spacing: -1.7px;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 19px;
    margin-bottom: 12px;
}

.gradient-text {
    background: linear-gradient(90deg, #60A5FA, #67E8F9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #A9B9D0;
    max-width: 680px;
    font-size: 14px;
    line-height: 1.8;
}

.panel {
    background: linear-gradient(145deg,
        rgba(20, 32, 52, .98),
        rgba(13, 23, 39, .98));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 23px;
    margin-bottom: 18px;
}

.section-label {
    color: #7DD3FC;
    font-size: 11px;
    letter-spacing: 1.6px;
    font-weight: 800;
    margin-bottom: 8px;
}

.panel-title {
    color: #F1F5F9;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 8px;
}

.panel-desc {
    color: #91A2BD;
    font-size: 13px;
    line-height: 1.65;
    margin-bottom: 18px;
}

.metric-card {
    padding: 21px;
    min-height: 125px;
    background: linear-gradient(145deg, #142139, #10192B);
    border: 1px solid var(--border);
    border-radius: 17px;
}

.metric-label {
    color: #91A2BD;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 29px;
    font-weight: 700;
    margin-top: 10px;
}

.metric-note {
    color: #60A5FA;
    font-size: 11px;
    margin-top: 5px;
}

.status-card {
    border-radius: 17px;
    padding: 21px;
    margin-top: 16px;
    margin-bottom: 12px;
}

.status-spam {
    background: rgba(127, 29, 29, .19);
    border: 1px solid rgba(248, 113, 113, .42);
}

.status-ham {
    background: rgba(6, 95, 70, .18);
    border: 1px solid rgba(52, 211, 153, .37);
}

.status-title {
    font-size: 22px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 7px;
}

.status-copy {
    font-size: 13px;
    color: #B8C5D9;
    line-height: 1.65;
}

.stTextArea textarea {
    background: #0A1323 !important;
    color: #F1F5F9 !important;
    border: 1px solid #293B56 !important;
    border-radius: 13px !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.7 !important;
}

.stTextArea textarea:focus {
    border-color: #60A5FA !important;
    box-shadow: 0 0 0 1px #60A5FA !important;
}

.stButton > button {
    width: 100%;
    min-height: 47px;
    border-radius: 12px;
    background: linear-gradient(100deg, #2563EB, #0891B2);
    color: white;
    border: 1px solid rgba(125, 211, 252, .25);
    font-weight: 700;
    transition: .2s ease;
}

.stButton > button:hover {
    border-color: #67E8F9;
    color: white;
    filter: brightness(1.13);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 9px;
    background: #0D1727;
    padding: 7px;
    border-radius: 13px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #A9B9D0;
    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    background: #1D3556 !important;
    color: #7DD3FC !important;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #3B82F6, #67E8F9);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    border-top: 1px solid var(--border);
    padding-top: 22px;
    margin-top: 35px;
    text-align: center;
    color: #7487A4;
    font-size: 11px;
    letter-spacing: .5px;
}

div[data-testid="stMetric"] {
    background: #111D30;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- HEADER ----------------

st.markdown("""
<div class="brand">
    <div class="brand-icon">🛡️</div>
    <div>
        <div class="brand-name">SMS SHIELD<span style="color:#67E8F9">.</span></div>
        <div class="brand-sub">INTELLIGENT MESSAGE SECURITY</div>
    </div>
</div>

<div class="hero">
    <div class="eyebrow">✦ MACHINE LEARNING • NLP • TEXT ANALYTICS</div>
    <div class="hero-title">
        Every message.<br>
        <span class="gradient-text">A smarter layer of security.</span>
    </div>
    <div class="hero-desc">
        Analyze SMS messages with a machine learning classifier trained
        on labeled messages. Explore predictions, test-set performance,
        and classification insights in one workspace.
    </div>
</div>
""", unsafe_allow_html=True)


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model files are missing. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

# ---------------- TOP METRICS ----------------

m1, m2, m3, m4 = st.columns(4)

metric_data = [
    (m1, "TEST ACCURACY", metrics["accuracy"], "Held-out test set"),
    (m2, "SPAM PRECISION", metrics["spam_precision"], "Positive predictions"),
    (m3, "SPAM RECALL", metrics["spam_recall"], "Spam messages detected"),
    (m4, "SPAM F1-SCORE", metrics["spam_f1"], "Precision-recall balance")
]

for col, label, value, note in metric_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value * 100:.2f}%</div>
            <div class="metric-note">↗ {note}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------- WORKSPACE TABS ----------------

tab_scan, tab_analytics, tab_about = st.tabs([
    "  ◉  Message Scanner  ",
    "  ▥  Model Analytics  ",
    "  ✦  About the Project  "
])

# ---------------- MESSAGE SCANNER ----------------

with tab_scan:
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("""
        <div class="panel">
            <div class="section-label">01 / MESSAGE INPUT</div>
            <div class="panel-title">Inspect a message</div>
            <div class="panel-desc">
                Paste an SMS below. The trained model will analyze its
                text patterns and classify it as Spam or Not Spam.
            </div>
        </div>
        """, unsafe_allow_html=True)

        sample_options = {
            "Write your own message": "",
            "Everyday conversation": (
                "Hey, are we meeting at college tomorrow? "
                "Please bring your notes."
            ),
            "Prize notification": (
                "Congratulations! You have won a free prize. "
                "Call now to claim your reward!"
            ),
            "Promotional offer": (
                "URGENT! You have been selected for a cash reward. "
                "Reply now to claim your offer."
            )
        }

        selected_sample = st.selectbox(
            "QUICK TEST SAMPLES",
            list(sample_options.keys()),
            key="sample_choice"
        )

        if (
            selected_sample != "Write your own message"
            and st.session_state.get("_last_sample") != selected_sample
        ):
            st.session_state["sms_input"] = sample_options[selected_sample]
            st.session_state["_last_sample"] = selected_sample

        message = st.text_area(
            "SMS MESSAGE",
            key="sms_input",
            height=175,
            placeholder="Type or paste your SMS here..."
        )

        analyze = st.button(
            "✦  ANALYZE MESSAGE",
            type="primary",
            use_container_width=True
        )

        if analyze:
            if not message.strip():
                st.warning("Enter a message before starting the analysis.")
            else:
                prediction = model.predict([message.strip()])[0]
                probabilities = model.predict_proba([message.strip()])[0]
                classes = list(model.named_steps["classifier"].classes_)
                spam_score = float(probabilities[classes.index("spam")])

                st.session_state["last_result"] = {
                    "message": message.strip(),
                    "prediction": prediction,
                    "spam_score": spam_score
                }

    with right:
        st.markdown("""
        <div class="panel">
            <div class="section-label">02 / ANALYSIS RESULT</div>
            <div class="panel-title">Security assessment</div>
            <div class="panel-desc">
                Your classification result will appear here after analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        result = st.session_state.get("last_result")

        if result:
            prediction = result["prediction"]
            spam_score = result["spam_score"]

            if prediction == "spam":
                st.markdown(f"""
                <div class="status-card status-spam">
                    <div style="font-size:29px">⚠️</div>
                    <div class="status-title" style="color:#FCA5A5">
                        SPAM DETECTED
                    </div>
                    <div class="status-copy">
                        The classifier identified patterns associated
                        with spam messages in its training data.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card status-ham">
                    <div style="font-size:29px">✓</div>
                    <div class="status-title" style="color:#6EE7B7">
                        NOT SPAM
                    </div>
                    <div class="status-copy">
                        The classifier categorized this message as
                        a legitimate message.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**Model-estimated spam score**")
            st.progress(spam_score)
            st.markdown(
                f"<div style='text-align:right;color:#67E8F9;"
                f"font-size:24px;font-weight:800'>{spam_score * 100:.2f}%"
                f"</div>",
                unsafe_allow_html=True
            )

            st.caption(
                "This is the model's estimated class probability, "
                "not a guarantee or necessarily a calibrated probability."
            )

            with st.expander("View analyzed message"):
                st.write(result["message"])

            if st.button("↻ Clear analysis"):
                st.session_state.pop("last_result", None)
                st.session_state["sms_input"] = ""
                st.session_state["_last_sample"] = "Write your own message"
                st.rerun()
        else:
            st.markdown("""
            <div class="panel" style="text-align:center;padding:45px 20px">
                <div style="font-size:45px">🔎</div>
                <div class="panel-title" style="margin-top:15px">
                    Awaiting input
                </div>
                <div class="panel-desc" style="margin-bottom:0">
                    Your result, spam score, and message assessment
                    will appear here.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <div class="section-label">HOW IT WORKS</div>
            <div style="color:#E2E8F0;font-size:13px;line-height:2">
                <span style="color:#67E8F9">01</span> &nbsp; Text vectorization<br>
                <span style="color:#67E8F9">02</span> &nbsp; Naive Bayes classification<br>
                <span style="color:#67E8F9">03</span> &nbsp; Prediction and score
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- MODEL ANALYTICS ----------------

with tab_analytics:
    st.markdown("""
    <div class="panel">
        <div class="section-label">MODEL INTELLIGENCE</div>
        <div class="panel-title">Performance overview</div>
        <div class="panel-desc">
            Evaluation results from the held-out test split.
            These metrics measure different aspects of model performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("#### Evaluation metrics")
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("Spam precision", f"{metrics['spam_precision'] * 100:.2f}%")
        st.metric("Spam recall", f"{metrics['spam_recall'] * 100:.2f}%")
        st.metric("Spam F1-score", f"{metrics['spam_f1'] * 100:.2f}%")

        st.info(
            "Precision measures how many messages predicted as spam "
            "were actually spam. Recall measures how many actual spam "
            "messages were detected."
        )

    with a2:
        st.markdown("#### Confusion matrix")

        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#10192B")
        ax.set_facecolor("#10192B")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.color_palette(
                ["#142139", "#2563EB", "#0891B2"], as_cmap=True
            ),
            xticklabels=["Not Spam", "Spam"],
            yticklabels=["Not Spam", "Spam"],
            linewidths=1,
            linecolor="#263750",
            cbar=False,
            annot_kws={"size": 16, "weight": "bold"},
            ax=ax
        )

        ax.set_xlabel("Predicted label", color="#B8C5D9", labelpad=12)
        ax.set_ylabel("Actual label", color="#B8C5D9", labelpad=12)
        ax.tick_params(colors="#B8C5D9")
        ax.set_title("Actual vs Predicted", color="#F1F5F9", pad=15)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("""
    <div class="panel">
        <div class="section-label">INTERPRETATION</div>
        <div class="panel-title">What the test results tell us</div>
        <div class="panel-desc">
            A strong overall accuracy does not mean every spam message
            is detected. Review recall alongside precision and F1-score
            when assessing a spam classifier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Correct not-spam", tn)
    q2.metric("False spam alerts", fp)
    q3.metric("Missed spam", fn)
    q4.metric("Spam detected", tp)

# ---------------- ABOUT ----------------

with tab_about:
    st.markdown("""
    <div class="panel">
        <div class="section-label">PROJECT OVERVIEW</div>
        <div class="panel-title">Inside SMS Shield</div>
        <div class="panel-desc">
            An educational machine learning application for classifying
            SMS messages using a labeled dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technology stack")

    t1, t2, t3 = st.columns(3)
    t1.markdown("""
    <div class="panel">
        <div style="font-size:28px">🐍</div>
        <div class="panel-title">Python</div>
        <div class="panel-desc">Application and data processing</div>
    </div>
    """, unsafe_allow_html=True)

    t2.markdown("""
    <div class="panel">
        <div style="font-size:28px">🧠</div>
        <div class="panel-title">TF-IDF + Naive Bayes</div>
        <div class="panel-desc">Text features and classification</div>
    </div>
    """, unsafe_allow_html=True)

    t3.markdown("""
    <div class="panel">
        <div style="font-size:28px">📊</div>
        <div class="panel-title">Streamlit</div>
        <div class="panel-desc">Interactive analytics interface</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Dataset and evaluation")

    st.write(
        f"Messages used after preprocessing: "
        f"**{metrics.get('total_messages', 'N/A')}**"
    )
    st.write(
        "The model was trained and evaluated on a stratified train/test "
        "split. Test-set metrics are displayed separately from individual "
        "message predictions."
    )

    st.warning(
        "This project is for educational use. Predictions can be wrong. "
        "Do not rely on the classifier alone to determine whether a "
        "message, link, or sender is safe."
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    SMS SHIELD &nbsp;•&nbsp; MACHINE LEARNING SECURITY LAB<br><br>
    BUILT WITH PYTHON · SCIKIT-LEARN · STREAMLIT
</div>
""", unsafe_allow_html=True)
