import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Academic Performance Analytics",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. THEME STATE (DARK / LIGHT)
# ==============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

THEMES = {
    "light": {
        "bg": "#f8f9fa",
        "card_bg": "#ffffff",
        "text_main": "#2D3748",
        "text_sub": "#718096",
        "border": "#e9ecef",
        "tab_bg": "#ffffff",
        "tab_selected_bg": "#f1f3f5",
        "chart_face": "#f8f9fa",
        "chart_plot": "#ffffff",
        "grid_color": "#cccccc",
        "tip_bg": "#ffffff",
    },
    "dark": {
        "bg": "#0E1117",
        "card_bg": "#1A1D24",
        "text_main": "#F0F2F6",
        "text_sub": "#A0AEC0",
        "border": "#2D3748",
        "tab_bg": "#1A1D24",
        "tab_selected_bg": "#262B36",
        "chart_face": "#1A1D24",
        "chart_plot": "#161A21",
        "grid_color": "#3A4150",
        "tip_bg": "#1A1D24",
    }
}

T = THEMES[st.session_state.theme]

# ==============================================================================
# 3. THEME-BASED CSS INJECTION
# ==============================================================================
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {T['bg']};
    }}
    h1, h2, h3, h4, h5, p, label, span, div {{
        color: {T['text_main']};
    }}
    .stButton>button {{
        background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(50,50,93,0.11), 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.15s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50,50,93,0.15);
        color: white !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {T['tab_bg']};
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-left: 16px;
        padding-right: 16px;
        border: 1px solid {T['border']};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {T['tab_selected_bg']};
        border-bottom: 2px solid #8E2DE2 !important;
        font-weight: bold;
    }}
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTimeInput input {{
        background-color: {T['card_bg']};
        color: {T['text_main']};
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. LOAD MODEL ARTIFACTS
# ==============================================================================
@st.cache_resource
def load_artifacts():
    with open('model_lr.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoders, le_target, scaler

try:
    model, label_encoders, le_target, scaler = load_artifacts()
except FileNotFoundError:
    st.error("❌ Oops! Model files (.pkl) are missing or not located in the same folder as app.py.")
    st.stop()

label_translation = {'Rendah': 'Low', 'Sedang': 'Medium', 'Tinggi': 'High'}
label_color = {'Low': '#E74C3C', 'Medium': '#F39C12', 'High': '#27AE60'}
label_bg_color_light = {'Low': '#FDEDEC', 'Medium': '#FEF9E7', 'High': '#EAFAF1'}
label_bg_color_dark = {'Low': '#3A2020', 'Medium': '#3A3320', 'High': '#1F3A2C'}
label_emoji = {'Low': '🚨', 'Medium': '⚡', 'High': '🏆'}

# Note: parental_education_level and attendance_percentage are no longer used
# because they were dropped from the model (very weak correlation with exam_score)
categorical_columns = ['gender', 'part_time_job', 'diet_quality',
                        'internet_quality', 'extracurricular_participation']

numerical_columns = ['age', 'study_hours_per_day', 'social_media_hours',
                      'netflix_hours', 'sleep_hours',
                      'exercise_frequency', 'mental_health_rating']

# ==============================================================================
# 5. HEADER + THEME TOGGLE
# ==============================================================================
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown(f"""
        <div style="padding: 10px 0 10px 0;">
            <h1 style="color: {T['text_main']}; font-weight: 800; font-size: 2.3rem; margin-bottom: 0.3rem;">
                🎓 Student Academic Performance Predictor
            </h1>
            <p style="color: {T['text_sub']}; font-size: 1.05rem; max-width: 600px; margin: 0;">
                Analyze how your daily routine — screen time, sleep duration, and study habits — affects your academic performance.
            </p>
        </div>
    """, unsafe_allow_html=True)
with header_col2:
    st.write("")
    icon_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    st.button(icon_label, on_click=toggle_theme, use_container_width=True)

st.markdown(f"<hr style='border:0.5px solid {T['border']}; margin-bottom:20px;'>", unsafe_allow_html=True)

# ==============================================================================
# 6. INPUT FORM (TABS)
# ==============================================================================
with st.form("prediction_form"):
    st.markdown(f"<h4 style='color:{T['text_main']}; margin-bottom:15px;'>📋 Fill in Your Profile & Habits</h4>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🕒 Routine & Media", "🧬 Background & Health"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            study_hours_per_day = st.number_input("📚 Effective Study Hours (Per Day)", min_value=0.0, max_value=24.0, value=3.5, step=0.5)
            social_media_hours = st.number_input("📱 Social Media Usage (Per Day)", min_value=0.0, max_value=24.0, value=2.5, step=0.5)
            netflix_hours = st.number_input("🎬 Netflix/Streaming Time (Per Day)", min_value=0.0, max_value=24.0, value=1.5, step=0.5)
        with col2:
            exercise_frequency = st.number_input("🏃 Exercise Frequency (Days/Week)", min_value=0, max_value=7, value=3, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-weight:600; color:{T['text_main']}; margin-bottom:5px;'>😴 Sleep Schedule</p>", unsafe_allow_html=True)

        st.markdown(f"<p style='color:{T['text_sub']}; font-size:13px; margin-bottom:5px;'>🌙 Night Sleep</p>", unsafe_allow_html=True)
        night_col1, night_col2, night_col3 = st.columns([1, 0.15, 1])
        with night_col1:
            night_sleep_start = st.time_input("From", value=time(22, 0), key="night_start", label_visibility="collapsed")
        with night_col2:
            st.markdown(f"<p style='text-align:center; margin-top:8px; color:{T['text_main']};'>—</p>", unsafe_allow_html=True)
        with night_col3:
            night_sleep_end = st.time_input("To", value=time(6, 0), key="night_end", label_visibility="collapsed")

        st.markdown(f"<p style='color:{T['text_sub']}; font-size:13px; margin-bottom:5px; margin-top:10px;'>🌤️ Nap (optional)</p>", unsafe_allow_html=True)
        nap_col1, nap_col2, nap_col3 = st.columns([1, 0.15, 1])
        with nap_col1:
            nap_start = st.time_input("From", value=time(13, 0), key="nap_start", label_visibility="collapsed")
        with nap_col2:
            st.markdown(f"<p style='text-align:center; margin-top:8px; color:{T['text_main']};'>—</p>", unsafe_allow_html=True)
        with nap_col3:
            nap_end = st.time_input("To", value=time(13, 0), key="nap_end", label_visibility="collapsed")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            age = st.number_input("👤 Age (Years)", min_value=16, max_value=30, value=20, step=1)
            gender = st.selectbox("🚻 Gender", ["Male", "Female", "Other"])
            part_time_job = st.selectbox("💼 Do You Have a Part-Time Job?", ["No", "Yes"])
            extracurricular_participation = st.selectbox("🧩 Participate in Extracurricular Activities?", ["No", "Yes"])
        with col4:
            diet_quality = st.selectbox("🥗 Diet Quality", ["Poor", "Fair", "Good"])
            internet_quality = st.selectbox("🌐 Internet Connection Quality", ["Poor", "Average", "Good"])
            mental_health_rating = st.selectbox("🧠 Self-Rated Mental Health (1-10)", list(range(1, 11)), index=4)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Calculate & Analyze Performance", use_container_width=True)

# ==============================================================================
# 7. HELPER: COMPUTE SLEEP DURATION FROM TIME RANGE
# ==============================================================================
def time_range_to_hours(start: time, end: time) -> float:
    """Converts a start/end time pair into a duration in hours, handling overnight ranges."""
    if start == end:
        return 0.0
    start_dt = datetime.combine(datetime.today(), start)
    end_dt = datetime.combine(datetime.today(), end)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return round((end_dt - start_dt).total_seconds() / 3600, 2)

# ==============================================================================
# 8. PREDICTION PROCESSING & RESULT VISUALIZATION
# ==============================================================================
if submitted:
    night_sleep_hours = time_range_to_hours(night_sleep_start, night_sleep_end)
    nap_hours = time_range_to_hours(nap_start, nap_end)
    sleep_hours = round(night_sleep_hours + nap_hours, 2)

    input_dict = {
        'age': age, 'gender': gender, 'study_hours_per_day': study_hours_per_day,
        'social_media_hours': social_media_hours, 'netflix_hours': netflix_hours,
        'part_time_job': part_time_job,
        'sleep_hours': sleep_hours, 'diet_quality': diet_quality,
        'exercise_frequency': exercise_frequency,
        'internet_quality': internet_quality, 'mental_health_rating': mental_health_rating,
        'extracurricular_participation': extracurricular_participation
    }

    input_df = pd.DataFrame([input_dict])

    for col in categorical_columns:
        le = label_encoders[col]
        if input_df[col].iloc[0] in le.classes_:
            input_df[col] = le.transform(input_df[col])
        else:
            input_df[col] = le.transform([le.classes_[0]])

    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    expected_columns = ['age', 'gender', 'study_hours_per_day', 'social_media_hours',
                         'netflix_hours', 'part_time_job',
                         'sleep_hours', 'diet_quality', 'exercise_frequency',
                         'internet_quality',
                         'mental_health_rating', 'extracurricular_participation']
    input_df = input_df[expected_columns]

    pred_encoded = model.predict(input_df)[0]
    pred_label_id = le_target.inverse_transform([pred_encoded])[0]
    pred_label_en = label_translation[pred_label_id]

    pred_proba = model.predict_proba(input_df)[0]
    proba_df = pd.DataFrame({
        'Category': [label_translation[c] for c in le_target.classes_],
        'Probability': pred_proba
    })

    st.markdown(f"<h3 style='margin-top:30px; color:{T['text_main']};'>📊 AI Analysis Result</h3>", unsafe_allow_html=True)

    color = label_color[pred_label_en]
    bg_color = (label_bg_color_dark if st.session_state.theme == "dark" else label_bg_color_light)[pred_label_en]
    emoji = label_emoji[pred_label_en]

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-left: 6px solid {color};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        ">
            <h3 style="color:{color}; margin:0; font-weight:700; display:flex; align-items:center; gap:10px;">
                {emoji} Performance Category: {pred_label_en}
            </h3>
            <p style="margin-top:10px; font-size:16px; color:{T['text_main']}; line-height:1.6;">
                Based on the Machine Learning algorithm's calculation of your daily habit patterns, your academic achievement level is projected to fall into the <strong>{pred_label_en}</strong> category.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color:{T['text_sub']}; font-size:13px; margin-top:-5px; margin-bottom:20px;'>"
        f"💤 Total Sleep Calculated: <strong>{sleep_hours} hours/day</strong> "
        f"(Night Sleep: {night_sleep_hours}h, Nap: {nap_hours}h)</p>",
        unsafe_allow_html=True
    )

    res_col1, res_col2 = st.columns([1.1, 0.9])

    with res_col1:
        st.markdown(f"<p style='font-weight:600; color:{T['text_main']}; margin-bottom:5px;'>Model Confidence Level</p>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 2.8))
        fig.patch.set_facecolor(T['chart_face'])
        ax.set_facecolor(T['chart_plot'])

        colors_bar = [label_color[c] for c in proba_df['Category']]
        bars = ax.barh(proba_df['Category'], proba_df['Probability'], color=colors_bar, height=0.55, edgecolor='none')

        ax.set_xlim(0, 1.15)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6, color=T['grid_color'])
        ax.set_axisbelow(True)
        ax.tick_params(colors=T['text_main'])

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%",
                    va='center', ha='left', fontsize=10, fontweight='bold', color=T['text_main'])

        plt.tight_layout()
        st.pyplot(fig)

    with res_col2:
        st.markdown(f"<p style='font-weight:600; color:{T['text_main']}; margin-bottom:5px;'>💡 Personalized Recommendations</p>", unsafe_allow_html=True)

        tips = []
        if study_hours_per_day < 3:
            tips.append("📚 <strong>Increase study time:</strong> Try adding at least 30-60 minutes per day consistently.")
        if social_media_hours > 4:
            tips.append("📱 <strong>Reduce screen time:</strong> Set app time limits on social media to stay focused.")
        if sleep_hours < 6:
            tips.append("😴 <strong>Improve your sleep schedule:</strong> Aim for 7 hours of sleep each night to keep your mind fresh for learning.")
        if mental_health_rating <= 4:
            tips.append("🧠 <strong>Self-Care:</strong> Reduce stress by talking to a counselor or close friend.")
        if exercise_frequency < 2:
            tips.append("🏃 <strong>Stay active:</strong> Make time for light exercise to boost physical fitness and focus stamina.")

        if not tips:
            st.success("🎉 Excellent! All your habits are well balanced and maintained. Keep up this performance!")
        else:
            for tip in tips:
                st.markdown(
                    f"<div style='background-color:{T['tip_bg']}; padding:10px 15px; border-radius:8px; "
                    f"margin-bottom:8px; border: 1px solid {T['border']}; font-size:14px; color:{T['text_main']};'>{tip}</div>",
                    unsafe_allow_html=True
                )

    st.markdown(f"<hr style='margin-top:30px; border:0.5px solid {T['border']};'>", unsafe_allow_html=True)
    st.caption(
        "⚠️ Disclaimer: This prediction is based purely on a Machine Learning model trained on a specific data "
        "sample and is intended only as a self-reflection tool, not an absolute academic diagnosis."
    )