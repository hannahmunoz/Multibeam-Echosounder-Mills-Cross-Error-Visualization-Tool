import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit

from math_utils import calculate_sidebar
from sidebar import set_sidebar_options


st.set_page_config(layout="wide", page_title="Mills Cross Error Visualization")

st.title("Multibeam Echosounder Mills Cross Visualization Tool")
st.markdown(
    "A visual aid to assess the impact of mechanical biases, dynamic IMU motion, and active beam steering on a flat seafloor baseline.")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Parameters")

set_sidebar_options()
calculate_sidebar()

st.subheader(f"Intersection Metrics for Queried Beam ({st.session_state["queried_angle"]}°)")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Along Dev (X)", f"{st.session_state["delta_x"]:.2f} m")
col2.metric("Across Dev (Y)", f"{st.session_state["delta_y"]:.2f} m")
col3.metric("Inside TX Fan?", st.session_state["tx_status"])
col4.metric("Inside RX Listening Area?", st.session_state["rx_status"])
col5.metric("Along Track Patch Width", f"{st.session_state["tx_x_width"]:.2f} m")
col6.metric("Sounding Patch Area", f"{st.session_state["patch_area"]:.2f} m²")