# ==========================================
# HEALTH RISK PREDICTOR DASHBOARD
# Advanced Streamlit ML Application - Premium Edition
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HealthRisk AI Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# HealthRisk AI Predictor v2.0\nDeveloped with cutting-edge ML technology"
    }
)

# ==========================================
# ADVANCED CUSTOM CSS STYLING - Light Green Theme
# ==========================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Root Variables - Light Green Theme */
    :root {
        --primary-gradient: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        --success-gradient: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        --danger-gradient: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        --light-bg: #f0f7f4;
        --card-bg: #ffffff;
        --border-color: #d5e8dc;
        --text-primary: #1a3c2a;
        --text-secondary: #5a7d6a;
        --accent-blue: #3498db;
        --accent-purple: #8e44ad;
        --accent-green: #27ae60;
        --accent-red: #e74c3c;
        --accent-orange: #f39c12;
        --shadow-color: rgba(46, 204, 113, 0.15);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f0f7f4 0%, #e8f5e9 50%, #f0f7f4 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: transparent;
    }
    
    /* Subtle Background Pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(46, 204, 113, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(39, 174, 96, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(52, 152, 219, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 3rem !important;
        background: linear-gradient(120deg, #27ae60, #2ecc71, #1abc9c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-bottom: 1.5rem !important;
        color: #1a3c2a !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        color: #1a3c2a !important;
    }
    
    /* Custom Card Styles */
    .custom-card {
        background: linear-gradient(145deg, #ffffff, #f8fdf9);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px var(--shadow-color);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .custom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
        border-radius: 20px 20px 0 0;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(46, 204, 113, 0.25);
        border-color: #27ae60;
    }
    
    /* Metric Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8fdf9);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-green);
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff, #f0f7f4);
        border-right: 1px solid var(--border-color);
    }
    
    /* Button Styles */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.5);
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
    }
    
    /* Slider Customization */
    .stSlider > div > div > div {
        background: var(--primary-gradient);
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: var(--primary-gradient);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: white !important;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    /* Alert/Info Boxes */
    .stAlert {
        border-radius: 16px;
        border: none;
        padding: 1.5rem;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animate-in {
        animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #27ae60;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: #27ae60 transparent transparent transparent !important;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    /* Glassmorphism Effect */
    .glass-effect {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(46, 204, 113, 0.2);
        border-radius: 20px;
    }
    
    /* Health Score Gauge */
    .gauge-container {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }
    
    /* Badge Styles */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background: var(--success-gradient);
        color: white;
    }
    
    .badge-danger {
        background: var(--danger-gradient);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
        color: white;
    }
    
    /* Notification Dot */
    .notification-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .dot-green {
        background: var(--accent-green);
        box-shadow: 0 0 10px var(--accent-green);
    }
    
    .dot-red {
        background: var(--accent-red);
        box-shadow: 0 0 10px var(--accent-red);
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL WITH CACHING
# ==========================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("health_risk_model.pkl")
        return model
    except:
        st.error("Model file not found. Please ensure 'health_risk_model.pkl' exists.")
        st.stop()

model = load_model()

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Predictor'

# ==========================================
# NAVIGATION
# ==========================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="background: linear-gradient(120deg, #27ae60, #2ecc71); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.8rem !important;">
        🏥 HealthAI
    </h2>
    <p style="color: #5a7d6a; font-size: 0.85rem; margin-top: -10px;">Intelligent Health Analytics</p>
</div>
""", unsafe_allow_html=True)

# Navigation
nav_options = ["🎯 Risk Predictor", "📊 Analytics Dashboard", "📈 Trend Analysis", "ℹ️ About"]
selected_nav = st.sidebar.radio("Navigation", nav_options, label_visibility="collapsed")

# ==========================================
# SIDEBAR HEADER WITH STATS
# ==========================================

st.sidebar.markdown("---")

# Quick Stats in Sidebar
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Scans", st.session_state.total_predictions, delta=None)
with col2:
    healthy_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 0)
    unhealthy_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 1)
    st.metric("Risk Detected", unhealthy_count, delta=None)

# Progress bar for healthy ratio
if st.session_state.total_predictions > 0:
    healthy_ratio = healthy_count / st.session_state.total_predictions
    st.sidebar.progress(healthy_ratio, text=f"Health Score: {healthy_ratio:.1%}")

st.sidebar.markdown("---")

# ==========================================
# PREDICTOR PAGE
# ==========================================

if "🎯 Risk Predictor" in selected_nav:
    
    # Dashboard Header with Animation
    st.markdown("""
    <div class="animate-in">
        <h1>🏥 HealthRisk AI Predictor</h1>
        <p style="font-size: 1.2rem; color: #5a7d6a; margin-top: -10px;">
            Advanced Machine Learning Healthcare Risk Assessment System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <span class="notification-dot dot-green"></span>
            <span style="color: #1a3c2a; font-weight: 600;">System Online</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <span style="color: #1a3c2a; font-weight: 600;">Model Accuracy</span>
            <br><span style="color: #27ae60; font-size: 1.2rem;">94.03%</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <span style="color: #1a3c2a; font-weight: 600;">Last Updated</span>
            <br><span style="color: #3498db; font-size: 1rem;">{datetime.now().strftime('%Y-%m-%d')}</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <span style="color: #1a3c2a; font-weight: 600;">Predictions</span>
            <br><span style="color: #8e44ad; font-size: 1.2rem;">{st.session_state.total_predictions}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SIDEBAR INPUTS WITH ENHANCED UI
    with st.sidebar:
        st.markdown("### 🩺 Health Parameters")
        st.markdown("Adjust the sliders to input patient health data")
        
        # Tabs for organized input
        input_tab1, input_tab2 = st.tabs(["📊 Vital Signs", "💊 Lifestyle"])
        
        with input_tab1:
            age = st.slider(
                "🎂 Age",
                min_value=0,
                max_value=100,
                value=25,
                help="Patient's age in years"
            )
            
            bmi = st.slider(
                "⚖️ BMI (Body Mass Index)",
                min_value=15.0,
                max_value=40.0,
                value=25.0,
                help="Body Mass Index - Normal range: 18.5-24.9"
            )
            
            blood_pressure = st.slider(
                "💗 Blood Pressure (Systolic)",
                min_value=80,
                max_value=220,
                value=120,
                help="Systolic blood pressure in mmHg"
            )
            
            cholesterol = st.slider(
                "🧬 Cholesterol",
                min_value=150,
                max_value=300,
                value=200,
                help="Total cholesterol level in mg/dL"
            )
            
            glucose = st.slider(
                "🩸 Glucose Level",
                min_value=70,
                max_value=200,
                value=100,
                help="Blood glucose level in mg/dL"
            )
            
            heart_rate = st.slider(
                "❤️ Heart Rate",
                min_value=50,
                max_value=120,
                value=72,
                help="Resting heart rate in beats per minute"
            )
        
        with input_tab2:
            sleep_hours = st.slider(
                "😴 Sleep Hours",
                min_value=0,
                max_value=14,
                value=7,
                help="Average hours of sleep per night"
            )
            
            exercise_hours = st.slider(
                "🏃 Exercise Hours",
                min_value=0,
                max_value=8,
                value=2,
                help="Hours of exercise per week"
            )
            
            water_intake = st.slider(
                "💧 Water Intake (Liters)",
                min_value=0,
                max_value=10,
                value=3,
                help="Daily water intake in liters"
            )
            
            stress_level = st.slider(
                "🧘 Stress Level",
                min_value=0,
                max_value=12,
                value=4,
                help="Stress level on a scale of 0-12"
            )
            
            st.markdown("#### Lifestyle Factors")
            smoking = st.selectbox("🚬 Smoking", ["No", "Yes"], index=1)
            alcohol = st.selectbox("🍺 Alcohol Consumption", ["No", "Yes"], index=1)
            diet = st.selectbox("🥗 Diet Quality", ["Poor", "Good"], index=1)
            
        # AI Recommendation Button
        st.markdown("---")
        predict_button = st.button("🔮 Generate AI Prediction", width='stretch', type="primary")
    
    # Main Content Area
    if predict_button or st.session_state.total_predictions > 0:
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'BMI': [bmi],
            'Blood_Pressure': [blood_pressure],
            'Cholesterol': [cholesterol],
            'Glucose_Level': [glucose],
            'Heart_Rate': [heart_rate],
            'Sleep_Hours': [sleep_hours],
            'Exercise_Hours': [exercise_hours],
            'Water_Intake': [water_intake],
            'Stress_Level': [stress_level],
            'Smoking': [1 if smoking == "Yes" else 0],
            'Alcohol': [1 if alcohol == "Yes" else 0],
            'Diet': [1 if diet == "Good" else 0],
            'MentalHealth': [1],
            'PhysicalActivity': [1],
            'MedicalHistory': [1],
            'Allergies': [1],
            'Diet_Type__Vegan': [0],
            'Diet_Type__Vegetarian': [1],
            'Blood_Group_AB': [0],
            'Blood_Group_B': [1],
            'Blood_Group_O': [0]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_probability = model.predict_proba(input_data)[0]
        
        # Store in history
        if predict_button:
            st.session_state.total_predictions += 1
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'age': age,
                'bmi': bmi,
                'prediction': prediction,
                'confidence': max(prediction_probability)
            })
        
        # Display Results in a stunning layout
        st.markdown("## 📊 Health Assessment Results")
        
        # Main Result Card
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="custom-card" style="animation: slideIn 0.6s ease;">
                    <div style="text-align: center;">
                        <span style="font-size: 4rem;">✅</span>
                        <h2 style="color: #27ae60 !important;">Healthy Status</h2>
                        <p style="color: #5a7d6a; font-size: 1.1rem;">
                            Based on the provided health parameters, the AI model predicts a 
                            <strong style="color: #27ae60;">Healthy</strong> status
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                confidence = prediction_probability[0] * 100
                
                st.markdown(f"""
                <div class="custom-card" style="text-align: center;">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Prediction Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="custom-card" style="animation: slideIn 0.6s ease;">
                    <div style="text-align: center;">
                        <span style="font-size: 4rem;" class="pulse">⚠️</span>
                        <h2 style="color: #e74c3c !important;">Higher Health Risk Detected</h2>
                        <p style="color: #5a7d6a; font-size: 1.1rem;">
                            The AI model has identified an <strong style="color: #e74c3c;">elevated health risk</strong> 
                            based on the provided parameters
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                confidence = prediction_probability[1] * 100
                
                st.markdown(f"""
                <div class="custom-card" style="text-align: center;">
                    <div class="metric-value" style="background: linear-gradient(135deg, #e74c3c, #c0392b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                        {confidence:.1f}%
                    </div>
                    <div class="metric-label">Risk Confidence</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Gauge Chart for Health Score
            health_score = prediction_probability[0] * 100
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Health Score", 'font': {'size': 24, 'color': '#1a3c2a'}},
                delta={'reference': 70, 'increasing': {'color': "#27ae60"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1a3c2a"},
                    'bar': {'color': "#27ae60"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#d5e8dc",
                    'steps': [
                        {'range': [0, 50], 'color': '#e74c3c'},
                        {'range': [50, 75], 'color': '#f39c12'},
                        {'range': [75, 100], 'color': '#27ae60'}
                    ],
                    'threshold': {
                        'line': {'color': "#1a3c2a", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#1a3c2a", 'family': "Inter"},
                height=300,
                margin=dict(t=50, b=0)
            )
            
            st.plotly_chart(fig_gauge, width='stretch')
        
        # Detailed Analysis Tabs
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Probability Analysis", "🎯 Risk Factors", "📊 Health Metrics", "💡 Recommendations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Donut Chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=['Healthy', 'At Risk'],
                    values=prediction_probability,
                    hole=.4,
                    marker_colors=['#27ae60', '#e74c3c'],
                    textinfo='label+percent',
                    textfont={'size': 14, 'color': '#1a3c2a'},
                    hovertemplate="%{label}: %{percent:.1%}<extra></extra>"
                )])
                
                fig_donut.update_layout(
                    title="Risk Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#1a3c2a'},
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=400
                )
                
                st.plotly_chart(fig_donut, width='stretch')
            
            with col2:
                # Probability Bar Chart
                prob_df = pd.DataFrame({
                    'Category': ['Healthy', 'Unhealthy'],
                    'Probability': prediction_probability,
                    'Color': ['#27ae60', '#e74c3c']
                })
                
                fig_prob = go.Figure(data=[
                    go.Bar(
                        x=prob_df['Category'],
                        y=prob_df['Probability'],
                        marker_color=prob_df['Color'],
                        text=[f'{p:.1%}' for p in prob_df['Probability']],
                        textposition='auto',
                        textfont={'size': 16, 'color': 'white'}
                    )
                ])
                
                fig_prob.update_layout(
                    title="Prediction Probabilities",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#1a3c2a'},
                    xaxis={'gridcolor': '#d5e8dc'},
                    yaxis={'gridcolor': '#d5e8dc', 'tickformat': '.0%'},
                    height=400
                )
                
                st.plotly_chart(fig_prob, width='stretch')
        
        with tab2:
            # Enhanced Feature Importance
            feature_importance = pd.DataFrame({
                'Feature': [
                    'BMI', 'Blood Pressure', 'Cholesterol', 'Stress Level',
                    'Glucose Level', 'Sleep Hours', 'Age', 'Heart Rate',
                    'Exercise Hours', 'Water Intake'
                ],
                'Importance': [0.216, 0.153, 0.100, 0.080, 0.073, 0.071, 0.070, 0.060, 0.055, 0.052],
                'Impact': ['High', 'High', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low']
            })
            
            feature_importance = feature_importance.sort_values('Importance', ascending=True)
            
            fig_importance = go.Figure(data=[
                go.Bar(
                    y=feature_importance['Feature'],
                    x=feature_importance['Importance'],
                    orientation='h',
                    marker=dict(
                        color=feature_importance['Importance'],
                        colorscale='Greens',
                        showscale=True,
                        colorbar=dict(title="Importance Score")
                    ),
                    text=[f'{i:.1%}' for i in feature_importance['Importance']],
                    textposition='outside',
                    textfont={'color': '#1a3c2a'}
                )
            ])
            
            fig_importance.update_layout(
                title="Feature Importance Analysis",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#1a3c2a'},
                xaxis={'gridcolor': '#d5e8dc', 'title': 'Importance Score'},
                yaxis={'gridcolor': '#d5e8dc'},
                height=500
            )
            
            st.plotly_chart(fig_importance, width='stretch')
            
            # Add interpretation
            st.markdown("""
            <div class="custom-card">
                <h4>🔍 Key Insights:</h4>
                <ul style="color: #5a7d6a;">
                    <li><strong style="color: #e74c3c;">BMI</strong> is the strongest predictor (21.6%)</li>
                    <li><strong style="color: #f39c12;">Blood Pressure & Cholesterol</strong> are significant cardiovascular indicators</li>
                    <li><strong style="color: #3498db;">Lifestyle factors</strong> (Sleep, Exercise, Water) show notable impact</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            # Radar Chart for Health Metrics
            categories = ['BMI', 'Blood Pressure', 'Cholesterol', 'Glucose', 
                         'Heart Rate', 'Sleep', 'Exercise', 'Water Intake', 'Stress']
            
            # Normalize values for radar chart
            values = [
                bmi/40, blood_pressure/220, cholesterol/300, glucose/200,
                heart_rate/120, sleep_hours/14, exercise_hours/8, water_intake/10, stress_level/12
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.3)',
                line=dict(color='#27ae60', width=2),
                name='Current Values'
            ))
            
            # Add reference values (optimal health)
            reference_values = [0.6, 0.55, 0.67, 0.5, 0.6, 0.57, 0.375, 0.4, 0.25]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=reference_values,
                theta=categories,
                fill='none',
                line=dict(color='#1abc9c', width=2, dash='dash'),
                name='Optimal Values'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor='#d5e8dc',
                        tickfont={'color': '#5a7d6a'}
                    ),
                    angularaxis=dict(
                        gridcolor='#d5e8dc',
                        tickfont={'color': '#1a3c2a'}
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    font={'color': '#1a3c2a'},
                    x=1,
                    y=1
                ),
                height=500
            )
            
            st.plotly_chart(fig_radar, width='stretch')
            
            # Heatmap of correlations
            st.subheader("Metric Correlation Matrix")
            
            metrics_data = pd.DataFrame({
                'BMI': [bmi, 22, 18, 30],
                'BP': [blood_pressure, 120, 90, 140],
                'Chol': [cholesterol, 180, 150, 240],
                'Glucose': [glucose, 90, 70, 130],
                'HR': [heart_rate, 70, 60, 100]
            }, index=['You', 'Optimal', 'Low', 'High'])
            
            fig_heatmap = ff.create_annotated_heatmap(
                z=metrics_data.values,
                x=list(metrics_data.columns),
                y=list(metrics_data.index),
                colorscale='Greens',
                showscale=True
            )
            
            fig_heatmap.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#1a3c2a'}
            )
            
            st.plotly_chart(fig_heatmap, width='stretch')
        
        with tab4:
            st.markdown("### 💡 AI-Generated Health Recommendations")
            
            recommendations = []
            
            if bmi > 25:
                recommendations.append({
                    'icon': '⚖️',
                    'title': 'Weight Management',
                    'description': 'Your BMI indicates overweight status. Consider consulting a nutritionist for a personalized diet plan.',
                    'priority': 'High'
                })
            
            if blood_pressure > 130:
                recommendations.append({
                    'icon': '💗',
                    'title': 'Blood Pressure Control',
                    'description': 'Elevated blood pressure detected. Reduce sodium intake and increase cardiovascular exercise.',
                    'priority': 'High'
                })
            
            if cholesterol > 200:
                recommendations.append({
                    'icon': '🧬',
                    'title': 'Cholesterol Management',
                    'description': 'High cholesterol levels. Incorporate more fiber-rich foods and omega-3 fatty acids.',
                    'priority': 'Medium'
                })
            
            if sleep_hours < 6:
                recommendations.append({
                    'icon': '😴',
                    'title': 'Sleep Optimization',
                    'description': 'Insufficient sleep detected. Aim for 7-9 hours of quality sleep per night.',
                    'priority': 'High'
                })
            
            if exercise_hours < 3:
                recommendations.append({
                    'icon': '🏃',
                    'title': 'Physical Activity',
                    'description': 'Low exercise levels. Try to get at least 150 minutes of moderate activity per week.',
                    'priority': 'Medium'
                })
            
            if stress_level > 7:
                recommendations.append({
                    'icon': '🧘',
                    'title': 'Stress Management',
                    'description': 'High stress levels detected. Consider meditation, yoga, or counseling.',
                    'priority': 'High'
                })
            
            if water_intake < 2:
                recommendations.append({
                    'icon': '💧',
                    'title': 'Hydration',
                    'description': 'Insufficient water intake. Aim for 2-3 liters of water daily.',
                    'priority': 'Medium'
                })
            
            # Display recommendations
            for rec in recommendations:
                priority_color = '#e74c3c' if rec['priority'] == 'High' else '#f39c12'
                
                st.markdown(f"""
                <div class="custom-card" style="margin: 0.5rem 0;">
                    <h4>{rec['icon']} {rec['title']} 
                        <span class="badge" style="background: {priority_color};">{rec['priority']}</span>
                    </h4>
                    <p style="color: #5a7d6a;">{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if not recommendations:
                st.success("🌟 Great job! Your health metrics are within optimal ranges. Maintain your healthy lifestyle!")
        
        # Input Data Display with Styling
        st.markdown("---")
        with st.expander("📄 View Complete Health Profile", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    input_data[['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose_Level', 'Heart_Rate']],
                    width='stretch'
                )
            
            with col2:
                st.dataframe(
                    input_data[['Sleep_Hours', 'Exercise_Hours', 'Water_Intake', 'Stress_Level', 'Smoking', 'Alcohol']],
                    width='stretch'
                )

# ==========================================
# ANALYTICS DASHBOARD PAGE
# ==========================================

elif "📊 Analytics Dashboard" in selected_nav:
    
    st.markdown("""
    <div class="animate-in">
        <h1>📊 Health Analytics Dashboard</h1>
        <p style="color: #5a7d6a;">Comprehensive analysis of health predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.total_predictions > 0:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="custom-card" style="text-align: center;">
                <div class="metric-value" style="font-size: 2rem;">{len(history_df)}</div>
                <div class="metric-label">Total Predictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            healthy_pct = (history_df['prediction'] == 0).mean() * 100
            st.markdown(f"""
            <div class="custom-card" style="text-align: center;">
                <div class="metric-value" style="font-size: 2rem; color: #27ae60;">{healthy_pct:.1f}%</div>
                <div class="metric-label">Healthy Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_confidence = history_df['confidence'].mean() * 100
            st.markdown(f"""
            <div class="custom-card" style="text-align: center;">
                <div class="metric-value" style="font-size: 2rem;">{avg_confidence:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_age = history_df['age'].mean()
            st.markdown(f"""
            <div class="custom-card" style="text-align: center;">
                <div class="metric-value" style="font-size: 2rem;">{avg_age:.0f}</div>
                <div class="metric-label">Avg Age</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution over time
            fig_timeline = px.scatter(
                history_df,
                x='timestamp',
                y='confidence',
                color='prediction',
                color_discrete_map={0: '#27ae60', 1: '#e74c3c'},
                title='Prediction Timeline',
                labels={'confidence': 'Confidence Score', 'timestamp': 'Time'}
            )
            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#1a3c2a'}
            )
            st.plotly_chart(fig_timeline, width='stretch')
        
        with col2:
            # Age vs BMI scatter
            fig_scatter = px.scatter(
                history_df,
                x='age',
                y='bmi',
                color='prediction',
                color_discrete_map={0: '#27ae60', 1: '#e74c3c'},
                title='Age vs BMI Distribution',
                labels={'age': 'Age', 'bmi': 'BMI'}
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#1a3c2a'}
            )
            st.plotly_chart(fig_scatter, width='stretch')
    else:
        st.info("📊 No predictions made yet. Use the Predictor page to generate health assessments.")

# ==========================================
# TREND ANALYSIS PAGE
# ==========================================

elif "📈 Trend Analysis" in selected_nav:
    
    st.markdown("""
    <div class="animate-in">
        <h1>📈 Health Trend Analysis</h1>
        <p style="color: #5a7d6a;">Monitor health patterns and trends over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.total_predictions > 1:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df.sort_values('timestamp')
        
        # Health Score Trend
        history_df['health_score'] = history_df.apply(
            lambda x: x['confidence'] if x['prediction'] == 0 else (1 - x['confidence']), 
            axis=1
        )
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['health_score'],
            mode='lines+markers',
            name='Health Score',
            line=dict(color='#27ae60', width=3),
            marker=dict(size=10, color=history_df['health_score'], colorscale='RdYlGn'),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ))
        
        fig_trend.update_layout(
            title='Health Score Trend Analysis',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#1a3c2a'},
            xaxis={'gridcolor': '#d5e8dc'},
            yaxis={'gridcolor': '#d5e8dc', 'title': 'Health Score'},
            height=500
        )
        
        st.plotly_chart(fig_trend, width='stretch')
        
        # BMI Trend
        fig_bmi = px.line(
            history_df,
            x='timestamp',
            y='bmi',
            title='BMI Trend Over Time',
            markers=True
        )
        fig_bmi.update_traces(line_color='#3498db')
        fig_bmi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#1a3c2a'}
        )
        
        st.plotly_chart(fig_bmi, width='stretch')
    else:
        st.info("📈 Need at least 2 predictions to show trends. Continue using the predictor!")

# ==========================================
# ABOUT PAGE
# ==========================================

else:
    
    st.markdown("""
    <div class="animate-in">
        <h1>ℹ️ About HealthRisk AI</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>🏥 Technology Stack</h3>
            <ul style="color: #5a7d6a; line-height: 2;">
                <li><strong style="color: #3498db;">Machine Learning:</strong> Random Forest Classifier with 94.03% accuracy</li>
                <li><strong style="color: #8e44ad;">Frontend:</strong> Streamlit with custom CSS animations</li>
                <li><strong style="color: #27ae60;">Visualization:</strong> Plotly interactive charts</li>
                <li><strong style="color: #f39c12;">Data Processing:</strong> Pandas, NumPy</li>
                <li><strong style="color: #e74c3c;">Model Storage:</strong> Joblib serialization</li>
            </ul>
        </div>
        
        <div class="custom-card" style="margin-top: 1rem;">
            <h3>🎯 Key Features</h3>
            <ul style="color: #5a7d6a; line-height: 2;">
                <li>✅ Real-time health risk prediction</li>
                <li>✅ Comprehensive health metrics analysis</li>
                <li>✅ AI-powered recommendations</li>
                <li>✅ Interactive data visualizations</li>
                <li>✅ Historical trend analysis</li>
                <li>✅ Probability confidence scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <h3>📊 Model Performance</h3>
            <div style="font-size: 3rem; font-weight: 800; background: linear-gradient(120deg, #27ae60, #2ecc71); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                94.03%
            </div>
            <p style="color: #5a7d6a;">Prediction Accuracy</p>
        </div>
        
        <div class="custom-card" style="text-align: center; margin-top: 1rem;">
            <h3>🔬 Features Analyzed</h3>
            <div style="font-size: 3rem; font-weight: 800; color: #3498db;">
                10+
            </div>
            <p style="color: #5a7d6a;">Health Parameters</p>
        </div>
        
        <div class="custom-card" style="text-align: center; margin-top: 1rem;">
            <h3>⚡ Response Time</h3>
            <div style="font-size: 3rem; font-weight: 800; color: #8e44ad;">
                <100ms
            </div>
            <p style="color: #5a7d6a;">Prediction Speed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #5a7d6a; padding: 2rem;">
        <h3 style="color: #1a3c2a;">Disclaimer</h3>
        <p>This tool is for educational and informational purposes only. 
        It should not be used as a substitute for professional medical advice, 
        diagnosis, or treatment. Always seek the advice of your physician or 
        other qualified health provider with any questions you may have regarding 
        a medical condition.</p>
        <br>
        <p>© 2024 HealthRisk AI Predictor | Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 2rem; color: #5a7d6a;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
        <span>🏥 HealthRisk AI</span>
        <span>|</span>
        <span>🧠 Powered by ML</span>
        <span>|</span>
        <span>📊 Real-time Analytics</span>
        <span>|</span>
        <span>🔒 Secure</span>
    </div>
    <p style="font-size: 0.85rem;">
        Developed with Streamlit + Machine Learning | Advanced Health Analytics Platform
    </p>
</div>
""", unsafe_allow_html=True)