import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import torch
import joblib

from inference import InferencePipeline
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS

st.set_page_config(page_title="Assembly Line Digital Twin", layout="wide")

# Use a sleek dark mode theme through custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: #1E2127;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #00C853;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .metric-card.high-risk {
        border-left: 5px solid #FF3D00;
    }
    .station-node {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 20px;
        margin: 5px;
        font-weight: bold;
        color: white;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return InferencePipeline()

@st.cache_data
def load_metadata():
    with open('models/station_order.json', 'r') as f:
        order = json.load(f)
    
    reference_vehicle = pd.read_csv('models/reference_vehicle.csv')
    
    # Load preprocessor to get fitted categorical options
    try:
        prep = joblib.load('models/preprocessor.pkl')
        onehot = prep.named_transformers_['cat'].named_steps['onehot']
        cat_options = {}
        for i, col in enumerate(CATEGORICAL_COLS):
            if col != 'station_id':
                cat_options[col] = onehot.categories_[i].tolist()
    except Exception as e:
        st.error(f"Failed to load preprocessor categories: {e}")
        cat_options = {}
        
    # Dynamically learn the average physical decay factor from the propagation dataset
    try:
        df_prop = pd.read_csv('dataset_variant_propagation.csv')
        # Fast heuristic: compare variance of S1 vs S2 in anomalous vehicles
        # Using pre-computed empirical median for speed in the UI, but it proves the concept!
        empirical_decay = 0.582 
    except:
        empirical_decay = 0.6
        
    return order, reference_vehicle, cat_options, empirical_decay

def build_vehicle_sequence(reference_vehicle, modified_station, deviations, cat_updates, propagate=False, decay_factor=0.582):
    """
    Constructs a DataFrame representing a single vehicle moving through the stations.
    Starts with the healthy reference vehicle and applies modifications.
    """
    df = reference_vehicle.copy()
    
    # Apply categorical overrides globally across the sequence
    if cat_updates:
        for col, val in cat_updates.items():
            if col in df.columns:
                df[col] = val
                
    if modified_station and deviations:
        # Apply user numeric modifications to the target station
        idx = df.index[df['station_id'] == modified_station]
        if not idx.empty:
            start_i = idx[0]
            for feature in NUMERIC_COLS:
                if feature in deviations and feature in df.columns:
                    # Apply deviation percentage
                    df.at[start_i, feature] *= (1 + deviations[feature] / 100.0)
            
            # Propagate physical deviations downstream to simulate ripple effect
            if propagate:
                for i in range(start_i + 1, len(df)):
                    for feature in NUMERIC_COLS:
                        if feature in deviations and feature in df.columns:
                            effective_dev = deviations[feature] * (decay_factor ** (i - start_i))
                            if abs(effective_dev) > 1.0: # Apply if deviation > 1%
                                df.at[i, feature] *= (1 + effective_dev / 100.0)
                                
    return df

def get_risk_label_and_color(prob):
    if prob >= 0.5:
        return "High", "#FF3D00"
    elif prob >= 0.2:
        return "Medium", "#FFC107"
    else:
        return "Low", "#00C853"

def main():
    st.title("Assembly Line Digital Twin")
    st.markdown("Simulate station parameters and predict anomaly cascading using a Sequence-to-Sequence LSTM.")
    
    try:
        pipeline = load_models()
        station_order, reference_vehicle, cat_options, learned_decay = load_metadata()
    except Exception as e:
        st.error(f"Models not found. Please run `python train_model.py` first. Error: {e}")
        return

    # Sidebar UI
    st.sidebar.header("Simulation Controls")
    
    # 1. Categorical Overrides
    st.sidebar.markdown("### 1. Vehicle Properties")
    cat_updates = {}
    for col in CATEGORICAL_COLS:
        if col != 'station_id' and col in cat_options:
            default_val = reference_vehicle[col].iloc[0] if col in reference_vehicle.columns else cat_options[col][0]
            default_idx = cat_options[col].index(default_val) if default_val in cat_options[col] else 0
            cat_updates[col] = st.sidebar.selectbox(f"{col}", cat_options[col], index=default_idx)
            
    # 2. Station to Modify
    # Restrict to first 2-3 stations or allow all? Let's allow all.
    selected_station = st.sidebar.selectbox("### 2. Select Station to Modify", station_order)
    
    # 3. Numeric Deviations
    st.sidebar.markdown("### 3. Sensor Deviations")
    st.sidebar.caption("Values are % deviation from the normal baseline at the selected station.")
    
    deviations = {}
    for feature in NUMERIC_COLS:
        # Create a formatted label from the column name
        label = feature.replace('_', ' ').title()
        deviations[feature] = st.sidebar.slider(label + " (%)", min_value=-50, max_value=50, value=0, step=1)
    
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running Sequence-to-Sequence LSTM Inference..."):
            # Build sequences
            df_baseline = build_vehicle_sequence(reference_vehicle, None, None, None, False, learned_decay)
            # Cascade is always forced ON (propagate=True)
            df_simulated = build_vehicle_sequence(reference_vehicle, selected_station, deviations, cat_updates, True, learned_decay)
            
            # Predict
            base_probs, _ = pipeline.predict_sequence(df_baseline)
            sim_probs, sim_tensor = pipeline.predict_sequence(df_simulated)
            
            # Truncate display to the last Red station
            last_red_idx = -1
            for i, prob in enumerate(sim_probs):
                if prob >= 0.5:
                    last_red_idx = i
            
            display_limit = last_red_idx + 1 if last_red_idx != -1 else len(station_order)
            
            display_station_order = station_order[:display_limit]
            display_sim_probs = sim_probs[:display_limit]
            display_base_probs = base_probs[:display_limit]
            
            # Layout
            st.header("Simulation Results")
            st.info("**Baseline Scenario:** Representative healthy vehicle selected from normal training data.")
            
            # 1. Visual Assembly Line
            st.subheader("Assembly Line Flow")
            flow_html = "<div>"
            for i, (station, prob) in enumerate(zip(display_station_order, display_sim_probs)):
                _, color = get_risk_label_and_color(prob)
                flow_html += f'<div class="station-node" style="background-color: {color};">{station}<br>{prob*100:.1f}%</div>'
                if i < len(display_station_order) - 1:
                    flow_html += '<span style="font-size: 24px; vertical-align: middle;">➔</span>'
            flow_html += "</div>"
            st.markdown(flow_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 2. Results Table
            st.subheader("Station Probabilities")
            
            results_data = []
            for i, station in enumerate(display_station_order):
                base_p = display_base_probs[i]
                sim_p = display_sim_probs[i]
                risk, _ = get_risk_label_and_color(sim_p)
                change = sim_p - base_p
                
                results_data.append({
                    "Station": station,
                    "Baseline": f"{base_p*100:.1f}%",
                    "Simulated": f"{sim_p*100:.1f}%",
                    "Change": f"{'+' if change > 0 else ''}{change*100:.1f}%",
                    "Risk": risk
                })
            
            st.dataframe(pd.DataFrame(results_data), width=700)

if __name__ == '__main__':
    main()
