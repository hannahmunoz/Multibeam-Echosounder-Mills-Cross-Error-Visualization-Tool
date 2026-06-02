import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit

from utils import get_sector_steering, get_rotation_matrix, generate_array_weights, solve_mills_cross_intersection, \
    project_to_flat_bottom, make_tx_ray, make_rx_ray


def calculate_sidebar_orientations():
    # --- MATH & GEOMETRY ---
    # True Mechanical Orientations (IMU Dynamic Motion + Static Mounting Biases)
    st.session_state["true_tx_roll"] = st.session_state["imu_roll"] + st.session_state["tx_roll_bias"]
    st.session_state["true_tx_pitch"] = st.session_state["imu_pitch"] + st.session_state["tx_pitch_bias"]
    st.session_state["true_tx_yaw"] = st.session_state["imu_yaw"] + st.session_state["tx_yaw_bias"]

    st.session_state["true_rx_roll"] = st.session_state["imu_roll"] + st.session_state["rx_roll_bias"]
    st.session_state["true_rx_pitch"] = st.session_state["imu_pitch"] + st.session_state["rx_pitch_bias"]
    st.session_state["true_rx_yaw"] = st.session_state["imu_yaw"] + st.session_state["rx_yaw_bias"]


def calculate_sidebar_stabilization():
    # Apply Active Roll Stabilization (Relies only on IMU values, capped at ±10°)
    st.session_state["array_relative_rx_angle"] = st.session_state["queried_angle"]
    if st.session_state["auto_roll"]:
        applied_rx_steer = np.clip(st.session_state["imu_roll"], -10.0, 10.0)
        st.session_state["array_relative_rx_angle"] -= applied_rx_steer

    # Dynamic Sector Steering (Pitch & Yaw Stabilization)
    swath_edges = np.linspace(-75.0, 75.0, st.session_state["num_sectors"] + 1)
    st.session_state["sector_limits"] = [(swath_edges[i], swath_edges[i + 1]) for i in
                                         range(st.session_state["num_sectors"])]


def calculate_steering():
    # Find which sector the red queried dot belongs to, so it uses the correct physical steering
    queried_sector_center = 0.0
    for s_start, s_end in st.session_state["sector_limits"]:
        if s_start <= st.session_state["queried_angle"] <= s_end:
            queried_sector_center = (s_start + s_end) / 2.0
            break

    # Convert variables for the Math Engine
    st.session_state["tx_steer_rad"] = get_sector_steering(queried_sector_center)
    st.session_state["tx_steer_angle"] = np.degrees(st.session_state["tx_steer_rad"])  # Preserve for fan geometry
    st.session_state["theta_rad"] = np.radians(st.session_state["array_relative_rx_angle"])


def calculate_secant_effect():
    # Calculate physical arrays based on a nominal 1500 m/s sound speed
    lambda_nom = 1500.0 / st.session_state["frequency"]
    st.session_state["lamda_nom"] = lambda_nom
    bw_factor = st.session_state["bw_factor"]
    st.session_state["L_tx"] = bw_factor * lambda_nom / np.radians(st.session_state["tx_beamwidth"])
    st.session_state["L_rx"] = bw_factor * lambda_nom / np.radians(st.session_state["rx_beamwidth"])

    # Calculate effective beamwidths based on the environmental sound speed slider
    wavelength = st.session_state["c_sound"] / st.session_state["frequency"]
    st.session_state["wavelength"] = wavelength
    st.session_state["tx_bw_rad"] = bw_factor * wavelength / st.session_state["L_tx"]
    st.session_state["rx_bw_rad"] = bw_factor * wavelength / st.session_state["L_rx"]

    # Apply the Secant Effect for the queried beam
    st.session_state["dynamic_tx_bw_rad"] = st.session_state["tx_bw_rad"] / np.cos(st.session_state["tx_steer_rad"])
    st.session_state["dynamic_rx_bw_rad"] = st.session_state["rx_bw_rad"] / np.cos(st.session_state["theta_rad"])


def calculate_rotation_matrix():
    st.session_state["R_tx_mech"] = get_rotation_matrix(st.session_state["true_tx_roll"],
                                                        st.session_state["true_tx_pitch"],
                                                        st.session_state["true_tx_yaw"])
    st.session_state["R_rx_mech"] = get_rotation_matrix(st.session_state["true_rx_roll"],
                                                        st.session_state["true_rx_pitch"],
                                                        st.session_state["true_rx_yaw"])

    # Ideal Matrices (Includes IMU motion and assumes no mounting biases)
    st.session_state["R_tx_ideal"] = get_rotation_matrix(st.session_state["imu_roll"], st.session_state["imu_pitch"],
                                                         st.session_state["imu_yaw"])
    st.session_state["R_rx_ideal"] = get_rotation_matrix(st.session_state["imu_roll"], st.session_state["imu_pitch"],
                                                         st.session_state["imu_yaw"])


def calculate_acoustic_directivity():
    # --- Acoustic Directivity and Hardware Math ---
    # Physical array elements are locked to the nominal half-wavelength (1500 m/s)
    d_spacing_nom = st.session_state["lamda_nom"] / 2.0
    d_lambda_eff = d_spacing_nom / st.session_state["wavelength"]  # Environmental spacing-to-wavelength ratio

    # Theoretical number of elements built into the hardware
    true_N_tx = int(np.ceil(st.session_state["L_tx"] / d_spacing_nom))
    true_N_rx = int(np.ceil(st.session_state["L_rx"] / d_spacing_nom))

    # Cap computational elements for Numba to maintain some semblance of UI speed
    comp_N_tx = max(1, min(true_N_tx, 300))
    comp_N_rx = max(1, min(true_N_rx, 300))

    # Pre-calculate distinct weights for TX and RX arrays
    st.session_state["tx_weights"] = generate_array_weights(comp_N_tx, shading=st.session_state["shading_type"])
    st.session_state["rx_weights"] = generate_array_weights(comp_N_rx, shading=st.session_state["shading_type"])

    # Calculate exact 3D nodes
    st.session_state["pt_calculated"] = solve_mills_cross_intersection(st.session_state["R_tx_ideal"],
                                                                       st.session_state["R_rx_ideal"],
                                                                       st.session_state["tx_steer_rad"],
                                                                       st.session_state["theta_rad"],
                                                                       st.session_state["depth"])
    st.session_state["pt_physical"] = solve_mills_cross_intersection(st.session_state["R_tx_mech"],
                                                                     st.session_state["R_rx_mech"],
                                                                     st.session_state["tx_steer_rad"],
                                                                     st.session_state["theta_rad"],
                                                                     st.session_state["depth"])


def calculate_tx_fan_geometry():
    # TX Fan Geometry Construction (Dynamic Multi-Sector Layout)
    tx_fwd_psi = st.session_state["tx_steer_rad"] + (st.session_state["dynamic_tx_bw_rad"] / 2)
    tx_aft_psi = st.session_state["tx_steer_rad"] - (st.session_state["dynamic_tx_bw_rad"] / 2)

    physical_tx_sectors = []
    calculated_tx_sectors = []

    for start_angle, end_angle in st.session_state["sector_limits"]:
        sector_center = (start_angle + end_angle) / 2.0
        sec_steer_rad = get_sector_steering(sector_center)

        # Secant beamwidth expansion for this specific sector
        sec_tx_bw_rad = st.session_state["tx_bw_rad"] / np.cos(sec_steer_rad)
        sec_fwd_psi = sec_steer_rad + (sec_tx_bw_rad / 2)
        sec_aft_psi = sec_steer_rad - (sec_tx_bw_rad / 2)

        # Physical Fan Sectors
        phys_fwd = [
            project_to_flat_bottom(np.dot(st.session_state["R_tx_mech"], make_tx_ray(t_s, sec_fwd_psi)).flatten()) for
            t_s in
            np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
        phys_aft = [
            project_to_flat_bottom(np.dot(st.session_state["R_tx_mech"], make_tx_ray(t_s, sec_aft_psi)).flatten()) for
            t_s in
            np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
        physical_tx_sectors.append(phys_fwd + list(reversed(phys_aft)))

        # Ideal Fan Sectors
        calc_fwd = [
            project_to_flat_bottom(np.dot(st.session_state["R_tx_ideal"], make_tx_ray(t_s, sec_fwd_psi)).flatten()) for
            t_s in
            np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
        calc_aft = [
            project_to_flat_bottom(np.dot(st.session_state["R_tx_ideal"], make_tx_ray(t_s, sec_aft_psi)).flatten()) for
            t_s in
            np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
        calculated_tx_sectors.append(calc_fwd + list(reversed(calc_aft)))


def calculate_rx_footprint_geometry():
    # RX Footprint Geometry Construction
    #required_acceptance_deg = abs(st.session_state["tx_steer_angle"]) + (st.session_state["tx_beamwidth"] / 2.0) + 2.0
    rx_acceptance_rad = np.radians(st.session_state["rx_fore_aft_bw"] / 2.0)

    half_rx_bw = st.session_state["dynamic_rx_bw_rad"] / 2
    theta_min = st.session_state["theta_rad"] - half_rx_bw
    theta_max = st.session_state["theta_rad"] + half_rx_bw

    rx_red_perimeter = []
    R_rx_mech = st.session_state["R_rx_mech"]
    rx_red_perimeter.extend(
        [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, rx_acceptance_rad)).flatten()) for t_s in
         np.linspace(theta_min, theta_max, 15)])
    rx_red_perimeter.extend([project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(theta_max, phi)).flatten()) for phi in
                             np.linspace(rx_acceptance_rad, -rx_acceptance_rad, 15)])
    rx_red_perimeter.extend(
        [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, -rx_acceptance_rad)).flatten()) for t_s in
         np.linspace(theta_max, theta_min, 15)])
    rx_red_perimeter.extend([project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(theta_min, phi)).flatten()) for phi in
                             np.linspace(-rx_acceptance_rad, rx_acceptance_rad, 15)])

    rx_full_perimeter = []
    rx_full_perimeter.extend(
        [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, rx_acceptance_rad)).flatten()) for t_s in
         np.linspace(-np.radians(77), np.radians(77), 50)])
    rx_full_perimeter.extend(
        [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(np.radians(77), phi)).flatten()) for phi in
         np.linspace(rx_acceptance_rad, -rx_acceptance_rad, 15)])
    rx_full_perimeter.extend(
        [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, -rx_acceptance_rad)).flatten()) for t_s in
         np.linspace(np.radians(77), -np.radians(77), 50)])
    rx_full_perimeter.extend(
        [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(-np.radians(77), phi)).flatten()) for phi in
         np.linspace(-rx_acceptance_rad, rx_acceptance_rad, 15)])

    rx_full_x = [p[0] for p in rx_full_perimeter] #TODO: unused?
    rx_full_y = [p[1] for p in rx_full_perimeter]
    rx_full_z = [p[2] for p in rx_full_perimeter]


def calculate_sidebar():
    calculate_sidebar_orientations()
    calculate_sidebar_stabilization()
    calculate_steering()
    calculate_secant_effect()
    calculate_rotation_matrix()
    calculate_acoustic_directivity()
    calculate_tx_fan_geometry()
    calculate_rx_footprint_geometry()
    print(st.session_state)