import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Tashkent Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f1923 100%);
    color: #e8edf2;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 700; color: #ffffff;
    line-height: 1.2; margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem; color: #7a9bb5; letter-spacing: 0.05em;
    text-transform: uppercase; font-weight: 500; margin-bottom: 1.5rem;
}
.hero-divider {
    width: 60px; height: 3px;
    background: linear-gradient(90deg, #f0a500, #f7c948);
    border-radius: 2px; margin-bottom: 2rem;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(240,165,0,0.2);
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #f0a500; }
.metric-label {
    font-size: 0.78rem; color: #7a9bb5;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;
}
.result-box {
    background: linear-gradient(135deg, rgba(240,165,0,0.12), rgba(247,201,72,0.06));
    border: 1.5px solid #f0a500; border-radius: 16px;
    padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; font-weight: 700; color: #f7c948;
}
.result-label {
    font-size: 0.85rem; color: #7a9bb5;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem;
}
.result-usd { font-size: 1.1rem; color: #b0c8de; margin-top: 0.5rem; font-weight: 500; }
.section-label {
    font-size: 0.72rem; font-weight: 600; color: #f0a500;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.8rem;
}
div.stButton > button {
    background: linear-gradient(135deg, #f0a500, #f7c948);
    color: #0f1923; font-weight: 700; font-size: 1rem;
    border: none; border-radius: 8px; padding: 0.7rem 2rem;
    width: 100%; transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.9; }
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(240,165,0,0.25) !important;
    color: #e8edf2 !important;
}
.info-note {
    background: rgba(122,155,181,0.1);
    border-left: 3px solid #7a9bb5;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #a0bcd4; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

DISTRICT_MEANS = {
    "Zangiota district":        72_000_000,
    "Qibray district":          68_000_000,
    "Yunusabad district":      145_000_000,
    "Chilanzar district":      110_000_000,
    "Mirzo Ulugbek district":  130_000_000,
    "Uchtepa district":         95_000_000,
    "Yakkasaray district":     155_000_000,
    "Shaykhantakhur district": 120_000_000,
    "Almazar district":        105_000_000,
    "Sergeli district":         88_000_000,
    "Yashnabad district":       90_000_000,
    "Yangikhayot district":     78_000_000,
    "Buka district":            62_000_000,
    "Parkent district":         65_000_000,
    "Akhangaran district":      58_000_000,
    "Piskent district":         55_000_000,
    "Chinaz district":          52_000_000,
    "Ortachirchiq district":    60_000_000,
    "Yangiyo'l district":       70_000_000,
    "Tashkent district":        75_000_000,
    "Bostanliq district":       48_000_000,
    "Lower Chirchiq district":  50_000_000,
    "Upper Chirchiq district":  54_000_000,
    "Bektemir district":        85_000_000,
}

HOUSE_TYPES = ["Block", "Wooden", "Brick", "Monolithic", "Panel"]
MARKET_TYPES = ["Secondary market", "Primary market"]

@st.cache_resource
def load_model():
    path = "rf_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), True
    return None, False

model, model_loaded = load_model()

def predict_price(district, house_type, market_type,
                  floor, total_floors, ceiling_h,
                  total_area, living_area, kitchen_area, rooms):

    market_enc = MARKET_TYPES.index(market_type)
    district_enc = DISTRICT_MEANS.get(district, 80_000_000)

    ht = {
        "house_type_Деревянный": 0,
        "house_type_Кирпичный":  0,
        "house_type_Монолитный": 0,
        "house_type_Панельный":  0,
    }
    house_type_map = {
        "Wooden":    "house_type_Деревянный",
        "Brick":     "house_type_Кирпичный",
        "Monolithic":"house_type_Монолитный",
        "Panel":     "house_type_Панельный",
    }
    if house_type in house_type_map:
        ht[house_type_map[house_type]] = 1

    row = pd.DataFrame([{
        "type_of_market":        market_enc,
        "number_of_rooms":       rooms,
        "total_living_area":     living_area,
        "total_area":            total_area,
        "floor":                 floor,
        "total_floors":          total_floors,
        "ceiling_height":        ceiling_h,
        "kitchen_area":          kitchen_area,
        "district_encoded":      district_enc,
        "house_type_Деревянный": ht["house_type_Деревянный"],
        "house_type_Кирпичный":  ht["house_type_Кирпичный"],
        "house_type_Монолитный": ht["house_type_Монолитный"],
        "house_type_Панельный":  ht["house_type_Панельный"],
    }])

    return model.predict(row)[0]

st.markdown('<div class="hero-title">🏠 Real Estate<br>Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Tashkent Region · Based on OLX Data</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
for col, (val, label, sub) in zip([col1, col2, col3, col4], [
    ("RF",  "Model",    "RandomForestRegressor"),
    ("200", "Trees",    "n_estimators"),
    ("13",  "Features", "Input columns"),
    ("✓",   "Status",   "Model loaded" if model_loaded else "Model not found"),
]):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}<br><span style="color:#4a7a99;font-size:0.7rem">{sub}</span></div>
        </div>""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ `rf_model.pkl` not found! Place the file in the same folder as `app.py`.")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
    district = st.selectbox("District", list(DISTRICT_MEANS.keys()), label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:1.2rem">🏗️ Property Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        house_type  = st.selectbox("House type", HOUSE_TYPES)
        market_type = st.selectbox("Market type", MARKET_TYPES)
        ceiling_h   = st.slider("Ceiling height (m)", 2.2, 3.5, 2.7, step=0.1)
    with c2:
        floor        = st.slider("Floor", 1, 25, 5)
        total_floors = st.slider("Total floors", 1, 25, 9)
        rooms        = st.slider("Number of rooms", 1, 12, 2)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">📐 Area (m²)</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        total_area   = st.number_input("Total",   20,  500, 65)
    with c4:
        living_area  = st.number_input("Living",  10,  400, 40)
    with c5:
        kitchen_area = st.number_input("Kitchen",  4,   80, 10)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("💡 Predict Price")

with right:
    st.markdown('<div class="section-label">📊 Result</div>', unsafe_allow_html=True)

    if living_area > total_area:
        st.warning("⚠️ Living area cannot be greater than total area.")
    elif floor > total_floors:
        st.warning("⚠️ Floor cannot be greater than total floors.")
    elif predict_btn:
        if not model_loaded:
            st.error("Model not loaded.")
        else:
            try:
                predicted = predict_price(
                    district, house_type, market_type,
                    floor, total_floors, ceiling_h,
                    total_area, living_area, kitchen_area, rooms
                )
                predicted = max(predicted, 5_000_000)
                usd  = predicted / 12_500
                low  = predicted * 0.90
                high = predicted * 1.10

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Estimated Price</div>
                    <div class="result-price">{predicted/1_000_000:.1f} <span style="font-size:1.4rem">mln UZS</span></div>
                    <div class="result-usd">≈ ${usd:,.0f} USD</div>
                    <div style="margin-top:1rem; font-size:0.8rem; color:#7a9bb5">
                        Range: {low/1_000_000:.1f} – {high/1_000_000:.1f} mln UZS
                    </div>
                    <div style="margin-top:0.5rem; font-size:0.72rem; color:#4a7a99">
                        RandomForestRegressor · rf_model.pkl
                    </div>
                </div>""", unsafe_allow_html=True)

                per_sqm = predicted / total_area
                st.markdown(f"""
                <div style="margin-top:1rem; display:flex; gap:0.8rem;">
                    <div class="metric-card" style="flex:1">
                        <div class="metric-value" style="font-size:1.2rem">{per_sqm/1_000_000:.2f}M</div>
                        <div class="metric-label">Price per m²</div>
                    </div>
                    <div class="metric-card" style="flex:1">
                        <div class="metric-value" style="font-size:1.2rem">{rooms} rooms</div>
                        <div class="metric-label">{total_area} m² · {floor}/{total_floors} floor</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div class="result-box" style="opacity:0.45">
            <div class="result-label">To see the result</div>
            <div class="result-price" style="font-size:1.6rem">Enter details</div>
            <div class="result-usd">and click the button</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-note">
        Model trained on OLX Tashkent region data.<br>
        Actual price may vary depending on market conditions.<br>
        USD rate: ~12,500 UZS
    </div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#3a5a72; padding:1rem 0; border-top:1px solid rgba(240,165,0,0.1)">
    RandomForestRegressor · scikit-learn · Tashkent Region OLX Dataset
</div>""", unsafe_allow_html=True)
