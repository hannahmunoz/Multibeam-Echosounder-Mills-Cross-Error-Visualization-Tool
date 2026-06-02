import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit

def calculate_directivity(v_geo, R_mech, steer_rad, is_tx):
    """Wrapper that dynamically routes TX or RX arrays to the fast Numba math."""
    v_local = np.dot(R_mech.T, v_geo)

    if is_tx:
        sin_theta = v_local[0]
        weights = st.session_state["tx_weights"]
    else:
        sin_theta = v_local[1]
        weights = st.session_state["rx_weights"]

    # Call the pre-compiled Numba function using the effective d_lambda
    return _numba_array_factor(sin_theta, steer_rad, d_lambda_eff, weights)


def make_tx_ray(theta_sweep, psi_steer):
    """Intersecting cone math for the TX array"""
    x = np.sin(psi_steer)
    y = np.sin(theta_sweep)
    z_sq = 1.0 - x ** 2 - y ** 2
    z = np.sqrt(max(z_sq, 1e-6))
    return np.array([x, y, z])


def make_rx_ray(theta_steer, phi_acceptance):
    """Intersecting cone math for the RX array"""
    x = np.sin(phi_acceptance)
    y = np.sin(theta_steer)
    z_sq = 1.0 - x ** 2 - y ** 2
    z = np.sqrt(max(z_sq, 1e-6))
    return np.array([x, y, z])


def project_to_flat_bottom(v_ray):
    if v_ray[2] < 1e-6:
        return np.array([v_ray[0] * 1e5, v_ray[1] * 1e5, depth])
    scale = depth / v_ray[2]
    return np.array([v_ray[0] * scale, v_ray[1] * scale, depth])

def calculate_absorption_fg(frequency_hz, T, S, D, pH, c_sound_user):
    """
    Calculates the acoustic absorption coefficient (alpha) in dB/m
    using the full Francois-Garrison (1982) model.
    """
    f = frequency_hz / 1000.0  # Convert Hz to kHz
    T_k = T + 273.15  # Convert Temperature to Kelvin

    # Use the user's manually adjusted sound speed to keep geometry consistent
    c = c_sound_user

    # Boric Acid Contribution (Dominant at low frequencies ~ <10 kHz)
    f1 = 2.8 * np.sqrt(S / 35.0) * (10.0 ** (4.0 - (1245.0 / T_k)))
    A1 = (8.86 / c) * (10.0 ** (0.78 * pH - 5.0))
    alpha_boric = (A1 * f1 * f ** 2) / (f ** 2 + f1 ** 2)

    # Magnesium Sulfate Contribution (Dominant at mid frequencies ~ 10-100 kHz)
    f2 = (8.17 * (10.0 ** (8.0 - (1990.0 / T_k)))) / (1.0 + 0.0018 * (S - 35.0))
    A2 = 21.44 * (S / c) * (1.0 + 0.025 * T)
    P2 = 1.0 - (1.37e-4 * D) + (6.2e-9 * D ** 2)
    alpha_mgso4 = (A2 * P2 * f2 * f ** 2) / (f ** 2 + f2 ** 2)

    # Pure Water Viscosity Contribution (Dominant at high frequencies ~>200 kHz)
    if T <= 20.0:
        A3 = 4.937e-4 - (2.59e-5 * T) + (9.11e-7 * T ** 2) - (1.50e-8 * T ** 3)
    else:
        A3 = 3.964e-4 - (1.146e-5 * T) + (1.45e-7 * T ** 2) - (6.50e-10 * T ** 3)

    P3 = 1.0 - (3.83e-5 * D) + (4.9e-10 * D ** 2)
    alpha_pure = A3 * P3 * f ** 2

    # Total attenuation is the sum of all three components (dB/km)
    alpha_db_km = alpha_boric + alpha_mgso4 + alpha_pure

    # Convert from dB/km to dB/m for local spatial calculations
    return alpha_db_km / 1000.0

# --- 3-Axis Rotation Matrix (Tait-Bryan Yaw-Pitch-Roll) ---
def get_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # Standard multiplication order: Z * Y * X
    return np.dot(R_z, np.dot(R_y, R_x))

# --- Mills Cross Intersection Solving ---
def solve_mills_cross_intersection(R_tx, R_rx, tx_steer_rad, rx_steer_rad, seafloor_depth):
    """Algebraically solves the 3D intersection using the OE874 Lab D Tp transformation method."""
    # Define ideal vectors and rotate them by their respective mechanical/IMU matrices
    tx_ideal = np.array([1.0, 0.0, 0.0])
    rx_ideal = np.array([0.0, 1.0, 0.0])

    tx_vec = np.dot(R_tx, tx_ideal)
    rx_vec = np.dot(R_rx, rx_ideal)

    # Create new orthonormal basis XYZ' (Tp matrix)
    xp = tx_vec / np.linalg.norm(tx_vec)
    zp = np.cross(tx_vec, rx_vec)
    zp = zp / np.linalg.norm(zp)
    yp = np.cross(zp, xp)
    yp = yp / np.linalg.norm(yp)
    Tp = np.column_stack((xp, yp, zp))

    # Calculate Non-Orthogonality angle (no_a)
    no_a = -np.arcsin(np.clip(np.dot(tx_vec, rx_vec), -1.0, 1.0))

    # Determine components in the local array frame
    y1 = np.sin(rx_steer_rad) / np.cos(no_a)
    y2 = np.sin(tx_steer_rad) * np.tan(no_a)

    rho_hor_sq = (y1 + y2) ** 2 + np.sin(tx_steer_rad) ** 2

    # Check if beams actually intersect (if rho > 1, they do not)
    if rho_hor_sq >= 1.0:
        return np.array([0.0, 0.0, 0.0])

    # Formulate beam vector in XYZ' and transform back to Geo space
    bv_p = np.array([np.sin(tx_steer_rad), y1 + y2, np.sqrt(1.0 - rho_hor_sq)])
    bv_geo = np.dot(Tp, bv_p)

    # Project to flat seafloor
    if bv_geo[2] < 1e-6:
        return np.array([0.0, 0.0, 0.0])

    scale = seafloor_depth / bv_geo[2]
    return bv_geo * scale

def generate_array_weights(N, shading="Hamming"):
    """Pre-calculates the amplitude weights for an N-element array."""
    n = np.arange(N)
    if shading == "Uniform":
        weights = np.ones(N)
    elif shading == "Hann":
        # 0.5 + 0.5 cosine weighting
        weights = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif shading == "Hamming":
        # 0.54 + 0.46 cosine weighting
        weights = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    else:
        weights = np.ones(N)

    # Normalize weights so the peak main-lobe amplitude is exactly 1.0
    return weights / np.sum(weights)


@njit(fastmath=True)
def _numba_array_factor(sin_theta, steer_rad, d_lambda, weights):
    """
    An attempt to use optimized discrete array factor calculation using machine code to reduce chugging.
    d_lambda = physical spacing (d) divided by wavelength.
    """
    # Calculate the baseline phase shift based on steering
    phase_shift = 2.0 * np.pi * d_lambda * (sin_theta - np.sin(steer_rad))

    sum_real = 0.0
    sum_imag = 0.0

    # Complex summation across all N elements
    for n in range(len(weights)):
        phase = n * phase_shift
        sum_real += weights[n] * np.cos(phase)
        sum_imag += weights[n] * np.sin(phase)

    # Return the absolute amplitude
    return np.sqrt(sum_real ** 2 + sum_imag ** 2)


def get_sector_steering(sector_center_angle):
    """Calculates the unique pitch and yaw steering required for a specific sector"""
    if st.session_state["auto_pitch"]:
        # Optimal pitch steering changes based on the across-track angle
        pitch_comp_rad = np.arcsin(np.sin(np.radians(-st.session_state["imu_pitch"])) * np.cos(np.radians(sector_center_angle)))
        steer_deg = np.degrees(pitch_comp_rad)

        # Clip to mechanical/electronic hardware limits
        steer_deg = np.clip(steer_deg, -10.0, 10.0)
    else:
        steer_deg = st.session_state["manual_tx_steer"]

    if st.session_state["auto_yaw"]:
        # Yaw steering math
        yaw_comp_rad = np.arctan(np.tan(np.radians(sector_center_angle)) * np.sin(np.radians(st.session_state["imu_yaw"])))
        steer_deg += np.degrees(yaw_comp_rad)

    # Tx sector steering limits
    outer_edge_deg = 0.0
    for s_start, s_end in st.session_state["sector_limits"]:
        if s_start <= sector_center_angle <= s_end:
            outer_edge_deg = max(abs(s_start), abs(s_end))
            break

    # Mathematical steering limit
    max_allowable_steer = max(0.0, 90.0 - outer_edge_deg - (st.session_state["tx_beamwidth"] / 2.0))
    steer_deg = np.clip(steer_deg, -max_allowable_steer, max_allowable_steer)

    # This is an attempt to restrict transmit to region within RX listening area. Likely a gross over-simplification of what is actually done.
    max_rx_catch_angle = max(0.0, (st.session_state["rx_fore_aft_bw"] / 2.0) - (st.session_state["tx_beamwidth"] / 2.0))
    steer_deg = np.clip(steer_deg, -max_rx_catch_angle, max_rx_catch_angle)

    return np.radians(steer_deg)

def generate_native_lobe(is_tx, color_scale, name):
    """Generates a 3D acoustic balloon mapped to the array's native mechanical axis."""

    # Dynamically center the grid on the steered beam
    if is_tx:
        # TX array is along x steered in pitch
        v_center = np.pi / 2 - tx_steer_rad
    else:
        # RX array is along Y steered in roll
        v_center = np.pi / 2 - theta_rad

    # Both fans sweep 180 degrees downwards
    u = np.linspace(0, np.pi, 150)

    # V sweeps +/- 55 degrees around the newly centered main lobe
    # Maybe revisit this?
    v = np.linspace(v_center - np.radians(45), v_center + np.radians(45), 200)

    U, V = np.meshgrid(u, v)

    if is_tx:
        X_unit = np.cos(V)
        Y_unit = np.sin(V) * np.cos(U)
        Z_unit = np.sin(V) * np.sin(U)
    else:
        X_unit = np.sin(V) * np.cos(U)
        Y_unit = np.cos(V)
        Z_unit = np.sin(V) * np.sin(U)

    R_linear = np.zeros_like(X_unit)

    for i in range(X_unit.shape[0]):
        for j in range(X_unit.shape[1]):
            v_geo = np.array([X_unit[i, j], Y_unit[i, j], Z_unit[i, j]])

            if is_tx:
                D = calculate_directivity(v_geo, R_tx_mech, tx_steer_rad, is_tx=True)
            else:
                D = calculate_directivity(v_geo, R_rx_mech, theta_rad, is_tx=False)

            R_linear[i, j] = np.abs(D)

    # Normalize and map to dB scale (-40dB Floor)
    R_dB = np.clip(20 * np.log10(R_linear + 1e-12), -40, 0)
    R_physical_radius = (R_dB + 40.0) / 40.0

    # Scale to push past the water column
    X_lobe = X_unit * R_physical_radius * lobe_scale
    Y_lobe = Y_unit * R_physical_radius * lobe_scale
    Z_lobe = Z_unit * R_physical_radius * lobe_scale

    return go.Surface(
        x=X_lobe, y=Y_lobe, z=Z_lobe,
        surfacecolor=R_dB,
        colorscale=color_scale,
        cmin=-40, cmax=0,
        name=name,
        showscale=False,
        opacity=0.6
    )
