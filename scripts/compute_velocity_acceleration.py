import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.signal import savgol_filter, medfilt

# ============================================================
# Step 1. Compute UTM coordinates
# ============================================================
def compute_utm_coordinates(df,
                            lon_col='locationLongitude(WGS84)',
                            lat_col='locationLatitude(WGS84)'):
    """Add UTM coordinates and relative trajectory columns to the DataFrame."""
    first_lon = df[lon_col].iloc[0]
    utm_zone = int((first_lon + 180) / 6) + 1
    is_northern = df[lat_col].iloc[0] >= 0

    transformer = Transformer.from_crs(
        crs_from="epsg:4326",  # WGS84
        crs_to=f"epsg:{32600 + utm_zone if is_northern else 32700 + utm_zone}",  # UTM zone
        always_xy=True
    )

    df['utm_x'], df['utm_y'] = transformer.transform(
        df[lon_col].values,
        df[lat_col].values
    )

    # Make trajectory relative to the first point
    x_ref, y_ref = df['utm_x'].iloc[0], df['utm_y'].iloc[0]
    df['x_rel'] = df['utm_x'] - x_ref
    df['y_rel'] = df['utm_y'] - y_ref

    return df


# ============================================================
# Step 2. Compute velocity and acceleration using Savitzky–Golay
# ============================================================
def compute_velocity_acceleration_robust(df,
                                         utm_x_col='utm_x',
                                         utm_y_col='utm_y',
                                         time_col='loggingTime(txt)',
                                         target_hz=50,
                                         sg_window_seconds=0.4,
                                         polyorder=3,
                                         median_kernel=5,
                                         clip_g=3.0):
    """
    Robust velocity & acceleration estimation:
      - resample to `target_hz` (default 50 Hz)
      - Savitzky-Golay derivatives for vx, vy, ax, ay
      - Tangential acceleration a_t = (v·a) / |v|
      - Median filter + optional clipping to remove spikes

    Units:
      velocity_mps    -> m/s
      acceleration_mps2 -> m/s^2 (tangential)
      velocity_mph, acceleration_mph_s provided as well
    """

    df = df.copy()
    # timestamps
    df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # remove exact duplicate timestamps (keep first)
    if df['timestamp'].duplicated().any():
        df = df.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)

    # resample to uniform time base
    dt_ms = int(1000 / target_hz)
    df = df.set_index('timestamp').resample(f'{dt_ms}ms').interpolate('time').reset_index()

    # time in seconds
    t = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds().values
    if len(t) < 3:
        raise ValueError("Not enough samples after resampling to compute derivatives.")

    # window length in samples (derived from seconds); must be odd and >= polyorder+2
    window_length = int(np.round(sg_window_seconds * target_hz))
    if window_length % 2 == 0:
        window_length += 1
    min_window = polyorder + 2
    if window_length < min_window:
        window_length = min_window if min_window % 2 == 1 else min_window + 1

    x = df[utm_x_col].values
    y = df[utm_y_col].values
    dt = np.mean(np.diff(t))

    # Savitzky-Golay derivatives
    vx = savgol_filter(x, window_length, polyorder, deriv=1, delta=dt)
    vy = savgol_filter(y, window_length, polyorder, deriv=1, delta=dt)
    ax = savgol_filter(x, window_length, polyorder, deriv=2, delta=dt)
    ay = savgol_filter(y, window_length, polyorder, deriv=2, delta=dt)

    # velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2)

    # tangential acceleration (along-track): (v . a) / |v|
    eps = 1e-8
    a_tangential = (vx * ax + vy * ay) / (v_mag + eps)

    # optional median filter to remove spikes (kernel length should be odd)
    if median_kernel and median_kernel > 1:
        if median_kernel % 2 == 0:
            median_kernel += 1
        a_tangential = medfilt(a_tangential, kernel_size=median_kernel)

    # Clip to physically plausible values (e.g., +-clip_g * 9.81 m/s^2)
    if clip_g is not None:
        max_accel = clip_g * 9.81
        a_tangential = np.clip(a_tangential, -max_accel, max_accel)

    # Attach to dataframe
    df['velocity_mps'] = v_mag
    df['acceleration_mps2'] = a_tangential

    # Convert units
    MPS_TO_MPH = 2.236936
    df['velocity_mph'] = df['velocity_mps'] * MPS_TO_MPH
    # acceleration in mph/s: (m/s^2) * (m/s -> mph) = m/s^2 * 2.236936
    df['acceleration_mph_s'] = df['acceleration_mps2'] * MPS_TO_MPH

    # summary
    print(f"Window len (s): {sg_window_seconds:.3f}, samples: {window_length}, dt ~ {dt:.4f}s")
    print(f"Peak speed: {df['velocity_mph'].max():.2f} mph")
    print(f"Peak accel (tangential): {df['acceleration_mph_s'].max():.2f} mph/s")
    print(f"Peak decel (tangential): {df['acceleration_mph_s'].min():.2f} mph/s")

    return df


# ============================================================
# Step 3. Plot velocity & acceleration
# ============================================================
def plot_velocity_acceleration(df,
                               time_col='timestamp',
                               velocity_col='velocity_mph',
                               acceleration_col='acceleration_mph_s',
                               save_path=None):
    """Plot velocity (mph) and acceleration (mph/s) vs time."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df[time_col], df[velocity_col], color='b', label='Velocity [mph]')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Velocity [mph]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df[time_col], df[acceleration_col], color='r', label='Acceleration [mph/s]', alpha=0.8)
    ax2.set_ylabel('Acceleration [mph/s]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Vehicle Velocity and Acceleration (Savitzky–Golay)')
    plt.grid(True)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.show()


# ============================================================
# Step 4. Example usage
# ============================================================
if __name__ == "__main__":
    root_folder = "/home/ubuntu/data/driving_data/EQE_wireless_charging_port/"
    file_path = os.path.join(root_folder, "2025-08-29_17_28_36_venrock_home_abrupt_brake.csv")

    df = pd.read_csv(file_path)

    #######################################################
    first_n_seconds = 60
    average_frequency = 100
    n_samples = int(first_n_seconds * average_frequency)
    df = df.iloc[:n_samples].copy()
    #######################################################

    # 1️⃣ Compute UTM and relative trajectory
    df = compute_utm_coordinates(df)

    # 2️⃣ Compute velocity (mph) and acceleration (mph/s) with Savitzky–Golay
    df = compute_velocity_acceleration_robust(df,
                                              time_col='loggingTime(txt)',
                                              target_hz=100,           # try 50 Hz first
                                              sg_window_seconds=0.3,  # 0.3 s window
                                              polyorder=3,
                                              median_kernel=5,
                                              clip_g=1.0)           

    # 3️⃣ Plot results
    output_plot = "/home/ubuntu/car_velocity_acceleration.png"
    plot_velocity_acceleration(df, save_path=output_plot)
