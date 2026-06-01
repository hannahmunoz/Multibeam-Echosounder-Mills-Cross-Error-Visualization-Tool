import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit

from sidebar import set_sidebar_options, calculate_sidebar


st.set_page_config(layout="wide", page_title="Mills Cross Error Visualization")

st.title("Multibeam Echosounder Mills Cross Visualization Tool")
st.markdown(
    "A visual aid to assess the impact of mechanical biases, dynamic IMU motion, and active beam steering on a flat seafloor baseline.")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Parameters")

set_sidebar_options()
calculate_sidebar()