import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit

from sidebar import load_sidebar_options
from utils import project_to_flat_bottom, make_tx_ray, calculate_directivity, make_rx_ray, get_rotation_matrix, \
    generate_array_weights, solve_mills_cross_intersection, calculate_absorption_fg, get_sector_steering, \
    generate_native_lobe

st.set_page_config(layout="wide", page_title="Mills Cross Error Visualization")

st.title("Multibeam Echosounder Mills Cross Visualization Tool")
st.markdown(
    "A visual aid to assess the impact of mechanical biases, dynamic IMU motion, and active beam steering on a flat seafloor baseline.")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Parameters")

load_sidebar_options()
