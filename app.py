import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import torch

from inference import InferencePipeline
from explainability import compute_feature_importance
from preprocessing import NUMERIC_COLS

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
    return order, reference_vehicle

def build_vehicle_sequence(reference_vehicle, modified_station, deviations):
    """
    Constructs a DataFrame representing a single vehicle moving through the stations.
    Starts with the healthy reference vehicle and applies modifications.
    """
    df = reference_vehicle.copy()
    
    if modified_station and deviations:
        # Apply user modifications to the target station
        idx = df.index[df['station_id'] == modified_station]
        if not idx.empty:
            i = idx[0]
            if 'cycle_time_sec' in df.columns: df.at[i, 'cycle_time_sec'] *= (1 + deviations['cycle_time_sec'] / 100.0)
            if 'torque_nm' in df.columns: df.at[i, 'torque_nm'] *= (1 + deviations['torque_nm'] / 100.0)
            if 'temperature_c' in df.columns: df.at[i, 'temperature_c'] *= (1 + deviations['temperature_c'] / 100.0)
            if 'vibration_rms' in df.columns: df.at[i, 'vibration_rms'] *= (1 + deviations['vibration_rms'] / 100.0)
            
    return df

def get_risk_label_and_color(prob):
    if prob >= 0.5:
        return "High", "#FF3D00"
    elif prob >= 0.2:
        return "Medium", "#FFC107"
    else:
        return "Low", "#00C853"

def main():
    st.title("🚗 Assembly Line Digital Twin")
    st.markdown("Simulate station parameters and predict anomaly cascading using a Sequence-to-Sequence LSTM.")
    
    try:
        pipeline = load_models()
        station_order, reference_vehicle = load_metadata()
    except Exception as e:
        st.error(f"Models not found. Please run `python train_model.py` first. Error: {e}")
        return

    # Sidebar UI
    st.sidebar.header("Simulation Controls")
    
    # Restrict to first 2-3 stations
    available_stations = station_order[:3]
    selected_station = st.sidebar.selectbox("1. Select Station to Modify", available_stations)
    
    st.sidebar.markdown("### 2. Modify Sensor Values")
    st.sidebar.caption("Values are % deviation from the normal baseline.")
    
    dev_cycle = st.sidebar.slider("Cycle Time Deviation (%)", min_value=-50, max_value=50, value=0, step=1)
    dev_torque = st.sidebar.slider("Torque Deviation (%)", min_value=-50, max_value=50, value=0, step=1)
    dev_temp = st.sidebar.slider("Temperature Deviation (%)", min_value=-50, max_value=50, value=0, step=1)
    dev_vib = st.sidebar.slider("Vibration Deviation (%)", min_value=-50, max_value=50, value=0, step=1)
    
    deviations = {
        'cycle_time_sec': dev_cycle,
        'torque_nm': dev_torque,
        'temperature_c': dev_temp,
        'vibration_rms': dev_vib
    }
    
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running Sequence-to-Sequence LSTM Inference..."):
            # Build sequences
            df_baseline = build_vehicle_sequence(reference_vehicle, None, None)
            df_simulated = build_vehicle_sequence(reference_vehicle, selected_station, deviations)
            
            # Predict
            base_probs, _ = pipeline.predict_sequence(df_baseline)
            sim_probs, sim_tensor = pipeline.predict_sequence(df_simulated)
            
            # Layout
            st.header("Simulation Results")
            st.info("ℹ️ **Baseline Scenario:** Representative healthy vehicle selected from normal training data.")
            
            # 1. Visual Assembly Line
            st.subheader("Assembly Line Flow")
            flow_html = "<div>"
            for i, (station, prob) in enumerate(zip(station_order, sim_probs)):
                _, color = get_risk_label_and_color(prob)
                flow_html += f'<div class="station-node" style="background-color: {color};">{station}<br>{prob*100:.1f}%</div>'
                if i < len(station_order) - 1:
                    flow_html += '<span style="font-size: 24px; vertical-align: middle;">➔</span>'
            flow_html += "</div>"
            st.markdown(flow_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 2. Results Table & Comparison
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Station Probabilities")
                
                results_data = []
                for i, station in enumerate(station_order):
                    base_p = base_probs[i]
                    sim_p = sim_probs[i]
                    risk, _ = get_risk_label_and_color(sim_p)
                    change = sim_p - base_p
                    
                    results_data.append({
                        "Station": station,
                        "Baseline": f"{base_p*100:.1f}%",
                        "Simulated": f"{sim_p*100:.1f}%",
                        "Change": f"{'+' if change > 0 else ''}{change*100:.1f}%",
                        "Risk": risk
                    })
                
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                
            # 3. Explainability
            with col2:
                st.subheader("Explainability")
                st.caption(f"Note: These are local feature influences derived from gradients, not guaranteed root causes.")
                
                station_idx = station_order.index(selected_station)
                top_features = compute_feature_importance(pipeline.model, sim_tensor, station_idx, pipeline.feature_names, NUMERIC_COLS)
                
                sim_p = sim_probs[station_idx]
                risk, _ = get_risk_label_and_color(sim_p)
                
                st.markdown(f"**{selected_station} — Anomaly Probability: {sim_p*100:.1f}%**")
                st.markdown("**Top Local Influencing Factors:**")
                for i, f in enumerate(top_features):
                    st.markdown(f"{i+1}. **{f['feature']}** — {f['direction']}")

if __name__ == '__main__':
    main()
