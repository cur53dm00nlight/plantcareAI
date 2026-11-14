import streamlit as st
import cv2
import numpy as np
import io
import html
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

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

div[data-testid="stFileUploader"] {
    background: #F0F7EF !important;
    border: 1px dashed #8EAD91 !important;
    border-radius: 14px;
    padding: 0.8rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #26734D !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F0F7EF !important;
    border: 1px dashed #8EAD91 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #183D30 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #26734D !important;
    border: 1px solid #26734D !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #FFFFFF !important;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

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

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stCaptionContainer"] {
    color: #183D30;
}

.hero-strip p {
    color: #DCEDE0 !important;
}

.stProgress > div > div > div > div {
    background-color: #39865A !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.stAlert p {
    color: #183D30 !important;
}

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


@st.cache_data(ttl=1800, show_spinner=False)
def geocode_location(place):
    """Find matching places with Open-Meteo's free geocoding service."""
    params = urlencode({"name": place, "count": 5, "language": "en", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
    request = Request(url, headers={"User-Agent": "PlantCareAI/1.0"})
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("results", [])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(latitude, longitude):
    """Fetch current conditions and a five-day forecast from Open-Meteo."""
    params = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "forecast_days": 5,
        "timezone": "auto"
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    request = Request(url, headers={"User-Agent": "PlantCareAI/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def weather_description(code):
    """Convert a WMO weather code into a readable description."""
    descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
        53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle",
        57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
        65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail"
    }
    return descriptions.get(code, "Conditions unavailable")


def make_weather_advisories(current, daily):
    """Create simple, rule-based general advisories from forecast measurements."""
    tips = []
    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    rain_next_day = daily.get("precipitation_sum", [0])[0] or 0
    rain_chance = daily.get("precipitation_probability_max", [0])[0] or 0

    if rain_chance >= 60 or rain_next_day >= 5:
        tips.append(("Rain expected", "Check drainage and postpone routine irrigation if sufficient rain is received."))
    elif rain_chance <= 20 and rain_next_day < 1:
        tips.append(("Low rain chance", "Check soil moisture regularly; irrigate only when the crop and soil need it."))

    if temperature is not None and temperature >= 35:
        tips.append(("High temperature", "Monitor plants for heat stress and check soil moisture. If irrigation is needed, cooler hours may reduce evaporation."))
    elif temperature is not None and temperature <= 10:
        tips.append(("Cool conditions", "Watch for cold-sensitive crops and follow locally recommended protection practices."))

    if humidity is not None and humidity >= 85:
        tips.append(("High humidity", "Inspect leaves regularly for signs of fungal disease and allow good airflow between plants."))

    if wind is not None and wind >= 25:
        tips.append(("Windy conditions", "Avoid activities that require calm weather, such as spraying, until wind conditions improve."))

    if not tips:
        tips.append(("General conditions", "Continue routine field monitoring and make irrigation decisions using soil moisture and crop requirements."))

    return tips

# --------------------------------------------------
# AGRICULTURAL MARKET PRICE FUNCTIONS
# --------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_prices(state, commodity, api_key):

    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 100,
        "filters[state.keyword]": state,
        "filters[commodity]": commodity
    }

    url = (
        f"https://api.data.gov.in/resource/{resource_id}?"
        f"{urlencode(params)}"
    )

    request = Request(
        url,
        headers={"User-Agent": "PlantCareAI/1.0"}
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return payload.get("records", [])

    except Exception as e:
        st.error(f"Could not fetch market prices: {e}")
        return []
# --------------------------------------------------
# AGRICULTURAL MARKET PRICE FUNCTIONS
# --------------------------------------------------


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_prices(state, commodity, api_key):
    """Fetch daily mandi prices from India's data.gov.in API."""

    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 100,
        "offset": 0,
        "filters[state.keyword]": state,
        "filters[commodity]": commodity
    }

    url = (
        f"https://api.data.gov.in/resource/{resource_id}?"
        f"{urlencode(params)}"
    )

    request = Request(
        url,
        headers={"User-Agent": "PlantCareAI/1.0"}
    )

    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload.get("records", [])

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
        ("↗", "How It Works"),
        ("☀", "Weather"),
        ("₹", "Market Prices"),
        ("📊", "Crop Profit Predictor")
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


# --------------------------------------------------
# WEATHER PAGE
# --------------------------------------------------

elif st.session_state.page == "Weather":

    st.markdown('<div class="main-title">Weather & Farming Advisory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Current conditions and a five-day forecast to help you plan field activities.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-strip">
            <div style="font-size:1.2rem; font-weight:800;">🌦️ Local weather insights</div>
            <p>Enter a town or city to retrieve live weather information from Open-Meteo.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("weather_search_form"):
        location_text = st.text_input(
            "Enter your city or town",
            placeholder="For example: Coimbatore, Chennai, Madurai"
        )
        search_weather = st.form_submit_button("Get weather", type="primary")

    if search_weather:
        if not location_text.strip():
            st.warning("Please enter a city or town.")
        else:
            try:
                with st.spinner("Finding your location..."):
                    matches = geocode_location(location_text.strip())
                if not matches:
                    st.warning("No matching location was found. Check the spelling or try a nearby larger town.")
                else:
                    st.session_state.weather_matches = matches
                    st.session_state.weather_search_text = location_text.strip()
                    st.session_state.weather_selected_index = 0
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                st.error(f"Could not look up this location. Check your internet connection and try again. Details: {error}")

    matches = st.session_state.get("weather_matches", [])
    if matches:
        labels = [
            ", ".join(part for part in [item.get("name"), item.get("admin1"), item.get("country")] if part)
            for item in matches
        ]
        selected_label = st.selectbox("Choose the matching location", labels, key="weather_location_choice")
        selected_index = labels.index(selected_label)
        place = matches[selected_index]

        try:
            with st.spinner("Fetching current conditions and the five-day forecast..."):
                weather = fetch_weather(place["latitude"], place["longitude"])

            current = weather.get("current", {})
            daily = weather.get("daily", {})
            units = weather.get("current_units", {})
            daily_units = weather.get("daily_units", {})

            st.markdown(f"### {html.escape(selected_label)}")
            st.caption(f"Weather data time: {current.get('time', 'Not provided')} · Time zone: {weather.get('timezone', 'local')}")

            if current:
                st.markdown("#### Current conditions")
                metric_columns = st.columns(4)
                metric_columns[0].metric("Temperature", f"{current.get('temperature_2m', '—')} {units.get('temperature_2m', '°C')}")
                metric_columns[1].metric("Humidity", f"{current.get('relative_humidity_2m', '—')} {units.get('relative_humidity_2m', '%')}")
                metric_columns[2].metric("Precipitation", f"{current.get('precipitation', '—')} {units.get('precipitation', 'mm')}")
                metric_columns[3].metric("Wind speed", f"{current.get('wind_speed_10m', '—')} {units.get('wind_speed_10m', 'km/h')}")
                st.info(f"Conditions: **{weather_description(current.get('weather_code'))}**")
            else:
                st.warning("Current conditions were not included in the response.")

            dates = daily.get("time", [])
            if dates:
                st.markdown("#### Five-day forecast")
                forecast_rows = []
                for index, date in enumerate(dates):
                    code = (daily.get("weather_code") or [None] * len(dates))[index]
                    high = (daily.get("temperature_2m_max") or [None] * len(dates))[index]
                    low = (daily.get("temperature_2m_min") or [None] * len(dates))[index]
                    rain = (daily.get("precipitation_sum") or [None] * len(dates))[index]
                    rain_chance = (daily.get("precipitation_probability_max") or [None] * len(dates))[index]
                    wind_max = (daily.get("wind_speed_10m_max") or [None] * len(dates))[index]
                    forecast_rows.append({
                        "Date": date,
                        "Conditions": weather_description(code),
                        "High (°C)": high,
                        "Low (°C)": low,
                        "Rain (mm)": rain,
                        "Rain chance (%)": rain_chance,
                        "Max wind (km/h)": wind_max
                    })
                st.dataframe(forecast_rows, use_container_width=True, hide_index=True)
            else:
                st.warning("The forecast was not included in the response.")

            if current and dates:
                st.markdown("#### General farming advisories")
                st.caption("These are simple rule-based suggestions based on the forecast, not crop-specific expert recommendations.")
                for heading, advice in make_weather_advisories(current, daily):
                    st.markdown(
                        f'<div class="panel" style="margin-bottom:0.65rem;"><strong>{html.escape(heading)}</strong><div class="mini-note">{html.escape(advice)}</div></div>',
                        unsafe_allow_html=True
                    )

            st.caption("Weather source: Open-Meteo (forecast and geocoding APIs). Forecasts can change; check local conditions before making farm decisions.")

        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
            st.error(f"Could not retrieve weather data right now. Please try again later. Details: {error}")
# --------------------------------------------------
# MARKET PRICES PAGE
# --------------------------------------------------

elif st.session_state.page == "Market Prices":
    st.title("Agricultural Market Prices")
    st.caption("Explore daily mandi prices for agricultural commodities across India.")

    api_key = st.secrets["DATA_GOV_API_KEY"]

    st.info("Market price data is provided by the Government of India's data.gov.in portal.")
    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox(
            "Select state",
            [
                "Tamil Nadu",
                "Kerala",
                "Karnataka",
                "Andhra Pradesh",
                "Telangana",
                "Maharashtra",
                "Punjab",
                "Uttar Pradesh"
            ]
        )

    with col2:
        commodity = st.selectbox(
            "Select commodity",
            [
                "Tomato",
                "Onion",
                "Potato",
                "Rice",
                "Wheat",
                "Maize",
                "Cotton",
                "Banana",
                "Coconut",
                "Groundnut"
            ]
        )
    st.markdown("---")

    if st.button("Fetch Market Prices", type="primary", use_container_width=True):
        with st.spinner("Fetching latest mandi prices..."):
            api_key = st.secrets["DATA_GOV_API_KEY"]
            records = fetch_market_prices(state, commodity, api_key)

        if records:
            st.success(f"Found {len(records)} market records.")

            price_columns = [
                "state",
                "district",
                "market",
                "commodity",
                "arrival_date",
                "min_price",
                "max_price",
                "modal_price"
            ]

            filtered_records = [
                {column: record.get(column) for column in price_columns}
                for record in records
            ]

            st.dataframe(
                filtered_records,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "No market price records found for this selection. "
                "Try another state or commodity."
            )
# --------------------------------------------------
# CROP PROFIT PREDICTOR PAGE
# --------------------------------------------------

elif st.session_state.page == "Crop Profit Predictor":
    st.title("Crop Profit Predictor")
    st.caption("Estimate your potential agricultural revenue, expenses and profit.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox(
            "Select crop",
            [
                "Rice",
                "Wheat",
                "Maize",
                "Tomato",
                "Onion",
                "Potato",
                "Cotton",
                "Groundnut",
                "Banana",
                "Coconut"
            ]
        )

    with col2:
        land_area = st.number_input(
            "Cultivated land area (acres)",
            min_value=0.1,
            value=1.0,
            step=0.1
        )
    st.markdown("---")
    st.subheader("Production and Selling Details")

    col1, col2 = st.columns(2)

    with col1:
        expected_yield = st.number_input(
            "Expected yield (quintals)",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    with col2:
        selling_price = st.number_input(
            "Expected selling price (₹ per quintal)",
            min_value=0.0,
            value=2500.0,
            step=100.0
        )
    st.markdown("---")
    st.subheader("Cultivation Expenses")

    cultivation_cost = st.number_input(
        "Total estimated cultivation cost (₹)",
        min_value=0.0,
        value=15000.0,
        step=500.0
    )
    st.markdown("---")

    if st.button("Calculate Profit", type="primary", use_container_width=True):
        total_revenue = expected_yield * selling_price
        total_expenses = cultivation_cost
        net_profit = total_revenue - total_expenses
        profit_per_acre = net_profit / land_area

        st.subheader("Profit Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
            st.metric("Total Expenses", f"₹{total_expenses:,.2f}")

        with col2:
            st.metric("Net Profit / Loss", f"₹{net_profit:,.2f}")
            st.metric("Profit / Acre", f"₹{profit_per_acre:,.2f}")

        if net_profit > 0:
            st.success("Your farm is projected to make a profit!")
        elif net_profit < 0:
            st.error("Your farm is projected to make a loss.")
        else:
            st.info("Your farm is projected to break even.")