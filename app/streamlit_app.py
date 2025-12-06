import streamlit as st
import json
import os
import sys

# Add the project root to sys.path so we can import from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agentic_dispatch_agent import run_agentic_dispatch

# Page config
st.set_page_config(
    page_title="Agentic Dispatch",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Agentic Dispatch & Curtailment Management")
st.markdown("""
This tool combines **Solar Forecasting**, **RAG Insights**, and **MILP Optimization** 
to recommend the best BESS dispatch strategy.
""")

# Sidebar for Plant Metadata
with st.sidebar:
    st.header("Plant Configuration")
    
    capacity_mwh = st.number_input(
        "BESS Capacity (MWh)", 
        min_value=1.0, 
        value=50.0,
        step=1.0,
        help="Total energy capacity of the battery system."
    )
    
    soc = st.slider(
        "Current SoC (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=35.0,
        step=1.0,
        help="Current State of Charge."
    ) / 100.0
    
    max_charge = st.number_input(
        "Max Charge Rate (MW)", 
        min_value=1.0, 
        value=50.0,
        step=1.0
    )
    
    max_discharge = st.number_input(
        "Max Discharge Rate (MW)", 
        min_value=1.0, 
        value=50.0,
        step=1.0
    )

    plant_meta = {
        "soc": soc,
        "capacity_mwh": capacity_mwh,
        "max_charge_mw": max_charge,
        "max_discharge_mw": max_discharge,
    }

# Main input area
query = st.text_area(
    "Operator Query",
    value="We expect high solar clipping at noon. How should we dispatch the battery?",
    height=100
)

if st.button("Run Analysis", type="primary"):
    with st.spinner("Running Agentic Dispatch... (Forecast → RAG → MILP)"):
        try:
            # Run the agent
            response = run_agentic_dispatch(query, plant_meta=plant_meta)
            
            # Display results
            st.success("Analysis Complete")
            st.markdown("### Agent Response")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.caption("Agentic RAG Prototype | Powered by OpenAI, Vertex AI, and OR-Tools")
