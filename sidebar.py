import streamlit as st

def set_sidebar_options():
    # Query beam information
    with st.sidebar.container(border=True):
        st.subheader("Interactive Beam Query")
        st.session_state["queried_angle"] = st.number_input("Query Specific Swath Angle (°)",
                                                            min_value=-75.0, max_value=75.0, value=45.0, step=1.0)

    with st.sidebar.expander("Environment", expanded=True):
        st.session_state["depth"] = st.number_input("Depth (m)", min_value=1.0, max_value=12000.0, value=100.0, step=10.0)
        st.session_state["c_sound"] = st.number_input("Sound Speed (m/s)", min_value=1400.0, max_value=1600.0, value=1500.0, step=1.0)
        st.session_state["water_temp"] = st.number_input("Water Temperature (°C)", min_value=-2.0, max_value=40.0, value=10.0, step=1.0)
        st.session_state["salinity"] = st.number_input("Salinity (ppt)", min_value=0.0, max_value=50.0, value=35.0, step=1.0)
        st.session_state["ph_level"] = st.number_input("pH", min_value=5.0, max_value=10.0, value=8.1, step=0.1)

    with st.sidebar.expander("Array Specifications", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state["frequency"] = st.number_input("Frequency (Hz)", min_value=1000.0, max_value=1000000.0, value=300000.0,
                                    step=10000.0)
        st.session_state["tx_beamwidth"] = c1.number_input("TX BW (Along-Track) (°)", value=0.5, step=0.1)
        st.session_state["rx_beamwidth"] = c1.number_input("RX BW (Across-Track) (°)", value=1.0, step=0.1)
        st.session_state["tx_across_fan_bw"] = c2.number_input("TX BW (Across-Track) (°)", value=150.0, step=1.0)
        st.session_state["rx_fore_aft_bw"] = c2.number_input("RX BW (Along-Track) (°)", value=30.0, step=1.0)

        st.session_state["target_swath_width"] = st.number_input("Target Swath Coverage (°)", min_value=10.0, max_value=150.0, value=120.0,
                                             step=1.0)
        st.session_state["num_sectors"] = st.selectbox("Number of TX Sectors", options=[1, 2, 3, 4, 5, 8], index=0)
        st.session_state["shading_type"] = st.selectbox("Array Shading", options=["Uniform", "Hann", "Hamming"], index=0)

    # Determine the beamwidth factor based on shading
    if st.session_state["shading_type"]  == "Uniform":
        st.session_state["bw_factor"] = 0.886
    elif st.session_state["shading_type"]  == "Hann":
        st.session_state["bw_factor"] = 1.20
    elif st.session_state["shading_type"]  == "Hamming":
        st.session_state["bw_factor"] = 1.30

    with st.sidebar.expander("Active Sonar Equation (Power & Noise)", expanded=True):
        st.session_state["source_level"] = st.number_input("Source Level (SL) [dB]", min_value=100.0, max_value=300.0, value=220.0,
                                       step=1.0)
        st.session_state["noise_level"] = st.number_input("Noise Level (NL) [dB]", min_value=10.0, max_value=120.0, value=50.0, step=1.0)
        st.session_state["bs_nadir"] = st.number_input("Baseline Scattering Strength [dB]", min_value=-60.0, max_value=0.0, value=-20.0,
                                   step=1.0)
        st.session_state["apply_tvg"] = st.checkbox("Apply TVG to Heatmap", value=True)

    # Dynamic Motion
    with st.sidebar.expander("Vessel Motion", expanded=True):
        c1, c2, c3 = st.columns(3)
        st.session_state["imu_roll"] = c1.number_input("Roll (°)", value=0.0, step=1.0)
        st.session_state["imu_pitch"] = c2.number_input("Pitch (°)", value=0.0, step=1.0)
        st.session_state["imu_yaw"] = c3.number_input("Yaw (°)", value=0.0, step=1.0)

    # Static Mounting Biases
    with st.sidebar.expander("Array Mounting Biases", expanded=True):
        st.markdown("**TX Array Biases**")
        c1, c2, c3 = st.columns(3)
        st.session_state["tx_roll_bias"] = c1.number_input("TX Roll (°)", value=0.0, step=1.0)
        st.session_state["tx_pitch_bias"] = c2.number_input("TX Pitch (°)", value=0.0, step=1.0)
        st.session_state["tx_yaw_bias"] = c3.number_input("TX Yaw (°)", value=0.0, step=1.0)

        st.markdown("**RX Array Biases**")
        c4, c5, c6 = st.columns(3)
        st.session_state["rx_roll_bias"] = c4.number_input("RX Roll (°)", value=0.0, step=1.0)
        st.session_state["rx_pitch_bias"] = c5.number_input("RX Pitch (°)", value=0.0, step=1.0)
        st.session_state["rx_yaw_bias"] = c6.number_input("RX Yaw (°)", value=0.0, step=1.0)

    # Active Stabilization
    with st.sidebar.expander("Active Stabilization & Steering", expanded=True):
        st.session_state["auto_roll"] = st.checkbox("Active Roll Stabilization (RX)", value=True)
        st.session_state["auto_pitch"] = st.checkbox("Active Pitch Stabilization (TX)", value=True)
        st.session_state["auto_yaw"] = st.checkbox("Active Yaw Stabilization (TX)", value=True)
        if not st.session_state["auto_pitch"]:
            st.session_state["manual_tx_steer"] = st.number_input("Manual TX Pitch Steer (°)", value=0.0, step=0.1)
        else:
            st.session_state["manual_tx_steer"] = 0.0

    with st.sidebar.expander("Acoustic Lobes", expanded=True):
        st.session_state[" show_tx_lobe"] = st.checkbox("Show TX Lobe (Blue)", value=False)
        if st.session_state.get("show_tx_lobe") and st.session_state["num_sectors"] > 1:
            st.warning(
                "Note: The 3D TX lobe display does not support simultaneous multi-sector visualization. It currently renders the active queried sector only.")
        st.session_state["show_rx_lobe"] = st.checkbox("Show RX Lobe (Red)", value=False)
        st.session_state["show_combined_lobe"] = st.checkbox("Show Combined Product Lobe", value=True)

