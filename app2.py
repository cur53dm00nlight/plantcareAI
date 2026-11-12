
import streamlit as st
import cv2
import numpy as np
import io
import html

from PIL import Image
from agri_india.predict import predict_proba, crops_and_labels
from diseaseinfo import get_disease_info


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PlantCare AI | Smart Agriculture",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --forest: #173F32;
    --forest-dark: #102F26;
    --green: #26734D;
    --green-light: #DCEBDD;
    --ink: #183D30;
    --muted: #64776B;
    --surface: #F8FBF6;
    --border: #D5E2D3;
}

/* Global typography */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 90% 4%, rgba(125, 165, 130, 0.20), transparent 28%),
        #D3DFD1;
}

.main .block-container {
    max-width: 1440px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #173F32 0%, #102F26 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #F4FAF4;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C8DCCF;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.18);
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    min-height: 46px;
    text-align: left;
    border-radius: 11px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: 0.18s ease;
}

[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    color: #EAF4EC;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.10);
}

[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    color: white;
    background: rgba(255,255,255,0.13);
    border-color: rgba(255,255,255,0.28);
}

[data-testid="stSidebar"] .stButton button[kind="primary"] {
    color: #FFFFFF !important;
    background: #24513F !important;
    border: 1px solid #70A987 !important;
}

[data-testid="stSidebar"] .stButton button[kind="primary"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    color: #FFFFFF !important;
    background: #32865B !important;
    border-color: #8BC19B !important;
}


/* Typography */

.main-title {
    color: var(--forest);
    font-size: clamp(2rem, 3vw, 2.65rem);
    font-weight: 800;
    letter-spacing: -1.4px;
    line-height: 1.12;
}

.subtitle {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.55rem;
    margin-bottom: 1.6rem;
}

.section-title {
    color: var(--ink);
    font-size: 1.12rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}

.condition-heading {
    color: #102F26 !important;
    font-size: 1.4rem;
    font-weight: 800;
    margin: 1rem 0 0.8rem 0;
}


/* HTML cards */

.panel {
    background: rgba(248, 251, 246, 0.94);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 8px 24px rgba(28, 66, 43, 0.055);
}

.result-card {
    background: linear-gradient(145deg, #FBFDF9 0%, #F0F7EF 100%);
    border: 1px solid #CFE0CF;
    border-radius: 18px;
    padding: 1.55rem;
    box-shadow: 0 8px 24px rgba(28, 66, 43, 0.06);
}

.result-label {
    color: #6D8072;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.25px;
    text-transform: uppercase;
}

.result-name {
    color: #174D35;
    font-size: clamp(1.55rem, 2.4vw, 2rem);
    font-weight: 800;
    line-height: 1.2;
    margin-top: 0.55rem;
    overflow-wrap: anywhere;
}

.confidence {
    color: #26734D;
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-top: 0.75rem;
}

.mini-note {
    color: #53695A;
    font-size: 0.86rem;
    line-height: 1.5;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 10px;
    color: #1F6844;
    background: #E0EFE0;
    font-weight: 800;
    margin-bottom: 0.75rem;
}

.hero-strip {
    background: linear-gradient(115deg, #173F32 0%, #286B49 100%);
    border-radius: 20px;
    padding: 1.5rem 1.7rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 28px rgba(23, 63, 50, 0.15);
}

.hero-strip p {
    color: #DCEDE0 !important;
    margin: 0.35rem 0 0 0;
}


/* Streamlit buttons */

.stButton button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

.stButton button[kind="primary"] {
    background: #26734D;
    border-color: #26734D;
    color: white;
}

.stButton button[kind="primary"]:hover {
    background: #174D35;
    border-color: #174D35;
    color: white;
}

/* Crop selection dropdown */

div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: #F0F7EF !important;
    border-radius: 12px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #F0F7EF !important;
    background-color: #F0F7EF !important;
    border: 1px solid #8EAD91 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #183D30 !important;
    -webkit-text-fill-color: #183D30 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #26734D !important;
    color: #26734D !important;
}

/* Dropdown options menu */

div[data-baseweb="popover"],
div[data-baseweb="popover"] [role="listbox"] {
    background: #F0F7EF !important;
}

div[data-baseweb="popover"] [role="option"] {
    background: #F0F7EF !important;
    color: #183D30 !important;
}

div[data-baseweb="popover"] [role="option"] * {
    color: #183D30 !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: #DCEBDD !important;
    color: #102F26 !important;
}
/* Image uploader outer container */

div[data-testid="stFileUploader"] {
    background: #F0F7EF !important;
    border: 1px dashed #8EAD91 !important;
    border-radius: 14px;
    padding: 0.8rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #26734D !important;
}

/* Actual drag-and-drop area */

[data-testid="stFileUploaderDropzone"] {
    background: #F0F7EF !important;
    border: 1px dashed #8EAD91 !important;
    border-radius: 12px !important;
}

/* Uploader text */

[data-testid="stFileUploaderDropzone"] * {
    color: #183D30 !important;
}

/* Upload button */

[data-testid="stFileUploaderDropzone"] button {
    background: #26734D !important;
    border: 1px solid #26734D !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #FFFFFF !important;
}


/* Uploaded image */

[data-testid="stImage"] img {
    border-radius: 14px;
}


/* Prediction breakdown dropdown */

[data-testid="stExpander"] {
    background: #F0F7EF !important;
    border: 1px solid #B8D2BB !important;
    border-radius: 12px !important;
    overflow: hidden;
    color: #183D30 !important;
}

[data-testid="stExpander"] summary {
    background: #173F32 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary:hover {
    background: #24513F !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #F0F7EF !important;
    color: #183D30 !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stCaptionContainer"] {
    color: #183D30 !important;
}


/* Condition information box */

div[data-testid="stAlert"],
div[data-testid="stAlert"] > div,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    background: #E5F0E3 !important;
    border-color: #C5DCC5 !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #183D30 !important;
}


/* Main results text */

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stCaptionContainer"] {
    color: #183D30;
}

/* Keep hero text light */

.hero-strip p {
    color: #DCEDE0 !important;
}


/* Progress bars */

.stProgress > div > div > div > div {
    background-color: #39865A !important;
}


/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.stAlert p {
    color: #183D30 !important;
}


/* Hide Streamlit branding */

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL INFORMATION
# --------------------------------------------------

@st.cache_resource
def get_crop_classes():
    return crops_and_labels()


try:
    crop_classes = get_crop_classes()
    crop_options = sorted(crop_classes.keys())
except Exception as e:
    st.error(f"Could not load AgriSmart model: {e}")
    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def display_name(value):
    """Convert model labels into readable names."""
    return value.replace("_", " ").replace("-", " ").title()


def process_image(uploaded_file):
    """Decode an uploaded image and prepare a display preview."""
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preview = cv2.resize(image, (128, 128))

    return image, preview


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"


with st.sidebar:

    st.markdown(
        """
        <div style="padding: 0.35rem 0 0.6rem 0;">
            <div style="font-size: 1.55rem; font-weight: 800; color: white;">
                🌿 PlantCare AI
            </div>
            <div style="font-size: 0.68rem; letter-spacing: 1.5px; color: #BBD5C4; margin-top: 0.25rem;">
                SMART AGRICULTURE TECHNOLOGY
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.caption("WORKSPACE")

    navigation = [
        ("⌂", "Home"),
        ("✦", "About"),
        ("↗", "How It Works")
    ]

    for symbol, label in navigation:

        button_label = f"{symbol}   {label}"

        if st.button(
            button_label,
            key=f"nav_{label}",
            type="primary" if st.session_state.page == label else "secondary",
            use_container_width=True
        ):
            st.session_state.page = label
            st.rerun()

    st.divider()

    st.markdown(
        """
        <div style="padding-top: 0.35rem;">
            <div style="font-size: 0.68rem; letter-spacing: 1.2px; color: #AFCDB9; font-weight: 700;">
                PROJECT
            </div>
            <div style="color: white; font-weight: 600; margin-top: 0.4rem;">
                Plant health analysis
            </div>
            <div style="color: #BBD5C4; font-size: 0.82rem; margin-top: 0.25rem;">
                Version 2.0 · School Project
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if st.session_state.page == "Home":

    st.markdown(
        '<div class="main-title">Plant Health Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-assisted leaf disease identification for smarter crop care.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-strip">
            <div style="font-size: 1.25rem; font-weight: 800;">
                A clearer view of plant health.
            </div>
            <p>Choose a crop, upload a leaf image, and review the model's prediction.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1], gap="large")


    # IMAGE INPUT

    with left:

        st.markdown(
            '<div class="section-title">01 &nbsp; Image & crop</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="mini-note">Start by selecting the crop shown in your photograph.</p>',
            unsafe_allow_html=True
        )

        selected_crop = st.selectbox(
            "Select crop",
            crop_options,
            format_func=lambda x: display_name(x)
        )

        uploaded_file = st.file_uploader(
            "Upload a leaf photograph",
            type=["jpg", "jpeg", "png", "jfif"],
            help="Use a clear, well-lit photograph where the leaf is visible."
        )

        processed_data = None

        if uploaded_file:

            processed_data = process_image(uploaded_file)

            if processed_data is not None:

                st.image(
                    processed_data[0],
                    caption="Uploaded leaf photograph",
                    use_container_width=True
                )

            else:
                st.error("Unable to process this image. Please try another file.")


    # ANALYSIS RESULTS

    with right:

        st.markdown(
            '<div class="section-title">02 &nbsp; Analysis results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="mini-note">Your prediction and the model score will appear here.</p>',
            unsafe_allow_html=True
        )

        if uploaded_file and processed_data is not None:

            try:

                original_image = Image.open(
                    io.BytesIO(uploaded_file.getvalue())
                ).convert("RGB")

                with st.spinner("Analysing leaf photograph..."):

                    predictions = predict_proba(
                        original_image,
                        crop=selected_crop
                    )

                ranked = sorted(
                    predictions.items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                if not ranked:

                    st.warning("The model did not return any predictions.")

                else:

                    predicted_label, confidence = ranked[0]

                    disease = predicted_label.split("::", 1)[-1]

                    safe_disease = html.escape(display_name(disease))

                    disease_details = get_disease_info(
                        selected_crop,
                        disease
                    )

                    safe_crop = html.escape(display_name(selected_crop))


                    # MAIN RESULT CARD

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Predicted condition</div>
                            <div class="result-name">{safe_disease}</div>
                            <div class="confidence">{confidence * 100:.2f}%</div>
                            <div class="result-label">Model confidence score</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "<div style='height: 0.8rem;'></div>",
                        unsafe_allow_html=True
                    )

                    st.success(f"Crop selected: {safe_crop}")


                    # PREDICTION BREAKDOWN

                    with st.expander("View prediction breakdown"):

                        for label, score in ranked[:5]:

                            disease_name = label.split("::", 1)[-1]

                            st.write(
                                f"**{display_name(disease_name)}**"
                            )

                            st.progress(float(score))

                            st.caption(
                                f"{score * 100:.2f}%"
                            )


                    # MODEL DISCLAIMER

                    st.caption(
                        "Important: this score is the model's output, not a guarantee "
                        "that the prediction is correct. Results should be checked "
                        "by a plant-health expert before taking action."
                    )


                    # CONDITION INFORMATION

                    st.markdown("---")

                    st.markdown(
                        '<div class="condition-heading">Condition Information</div>',
                        unsafe_allow_html=True
                    )

                    st.info(disease_details["about"])


                    if disease_details["type"] == "healthy":

                        st.markdown("**Healthy leaf characteristics**")

                        for item in disease_details["healthy_characteristics"]:

                            st.markdown(f"- {item}")

                    else:

                        sections = [
                            ("Visible symptoms", "symptoms"),
                            ("Possible causes", "causes"),
                            ("Precautions", "precautions")
                        ]

                        for title, key in sections:

                            if disease_details[key]:

                                st.markdown(f"**{title}**")

                                for item in disease_details[key]:

                                    st.markdown(f"- {item}")


            except Exception as e:

                st.error(f"Could not analyse this image: {e}")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif st.session_state.page == "About":

    st.markdown(
        '<div class="main-title">About PlantCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Technology supporting smarter agriculture.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-strip">
            <div style="font-size:1.25rem; font-weight:800;">
                Understanding plant health through AI
            </div>
            <p>PlantCare AI helps users explore possible leaf conditions using image classification.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")


    with col1:

        st.markdown(
            """
            <div class="panel" style="height:100%;">
                <div class="step-number">01</div>
                <div class="section-title">What is PlantCare AI?</div>
                <p class="mini-note">
                    PlantCare AI is an artificial intelligence application designed
                    to identify potential plant diseases from leaf photographs.
                    Users select the crop, upload an image, and receive a predicted condition.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="panel" style="height:100%;">
                <div class="step-number">02</div>
                <div class="section-title">The technology</div>
                <p class="mini-note">
                    The application integrates the AgriSmart India pre-trained
                    multi-crop classification model. The model analyses the uploaded
                    photograph and ranks possible leaf conditions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "<div style='height: 1rem;'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Acknowledgement</div>
            <p class="mini-note">
                PlantCare AI uses the AgriSmart India model. The original model
                developers and project repository are credited below.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.link_button(
        "View AgriSmart AI repository",
        "https://github.com/wpzvqrs8/SIH_2026"
    )

    st.warning(
        "PlantCare AI provides preliminary AI predictions. It is not a substitute "
        "for professional agricultural advice."
    )


# --------------------------------------------------
# HOW IT WORKS PAGE
# --------------------------------------------------

elif st.session_state.page == "How It Works":

    st.markdown(
        '<div class="main-title">How It Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">A simple process powered by artificial intelligence.</div>',
        unsafe_allow_html=True
    )

    steps = [
        (
            "01",
            "Crop selection",
            "Select the crop shown in the photograph so the model analyses the relevant group of conditions."
        ),
        (
            "02",
            "Image upload",
            "Upload a clear photograph of the plant leaf in JPG, JPEG, PNG, or JFIF format."
        ),
        (
            "03",
            "AI processing",
            "The image is passed to the pre-trained classification model, which evaluates possible conditions."
        ),
        (
            "04",
            "Results",
            "Review the predicted condition, model confidence score, and the top alternative predictions."
        )
    ]

    for number, heading, description in steps:

        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:0.9rem;">
                <div style="display:flex; gap:1rem; align-items:flex-start;">
                    <div class="step-number">{number}</div>
                    <div>
                        <div style="color:#174D35; font-size:1.1rem; font-weight:800; margin:0.15rem 0 0.35rem 0;">
                            {html.escape(heading)}
                        </div>
                        <div class="mini-note">{html.escape(description)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "The confidence score represents the model's output, not a guarantee of "
        "accuracy. Confirm important decisions with a qualified plant-health expert."
    )
