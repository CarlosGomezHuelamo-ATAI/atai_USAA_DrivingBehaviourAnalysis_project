import pdb
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pyproj import Transformer

# ============================================================
# ------------------ Quaternion math helpers -----------------
# ============================================================

def quat_to_rot_matrix(qx, qy, qz, qw):
    """Convert quaternion (x,y,z,w) to rotation matrix (device → earth)."""
    x, y, z, w = qx, qy, qz, qw
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return np.eye(3)
    x /= n; y /= n; z /= n; w /= n
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x*x + y*y)]
    ])
    return R


def rotate_device_to_earth(acc_device, qx, qy, qz, qw):
    R = quat_to_rot_matrix(qx, qy, qz, qw)
    return R.dot(acc_device)


def earth_to_car(acc_earth, heading_deg):
    """Rotate earth-frame vector into car frame using heading_deg (clockwise from North)."""
    theta = math.radians(heading_deg if not np.isnan(heading_deg) else 0.0)
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]])
    return Rz.dot(acc_earth)


# ============================================================
# ------------- Compute car-frame acceleration ---------------
# ============================================================

def compute_car_acceleration(df):
    """Compute car-frame acceleration (m/s²) from smartphone quaternion + heading."""
    G_TO_MS2 = 9.80665
    results = []

    for _, row in df.iterrows():
        acc_device = np.array([
            row.get("motionUserAccelerationX(G)", 0),
            row.get("motionUserAccelerationY(G)", 0),
            row.get("motionUserAccelerationZ(G)", 0)
        ]) * G_TO_MS2

        qx = row.get("motionQuaternionX(R)", 0)
        qy = row.get("motionQuaternionY(R)", 0)
        qz = row.get("motionQuaternionZ(R)", 0)
        qw = row.get("motionQuaternionW(R)", 1)

        acc_earth = rotate_device_to_earth(acc_device, qx, qy, qz, qw)

        heading = (
            row.get("locationTrueHeading(°)")
            if not pd.isna(row.get("locationTrueHeading(°)", np.nan))
            else row.get("motionHeading(°)", 0)
        )

        acc_car = earth_to_car(acc_earth, heading)

        results.append({
            "motionTimestamp_sinceReboot(s)": row.get("motionTimestamp_sinceReboot(s)", np.nan),
            "acc_car_x(ms2)": acc_car[0],
            "acc_car_y(ms2)": acc_car[1],
            "acc_car_z(ms2)": acc_car[2],
            "acc_car_mag(ms2)": np.linalg.norm(acc_car),
        })

    return pd.DataFrame(results)


# ============================================================
# -------------------- Helper functions ----------------------
# ============================================================

def smooth_signal(x, window=51, poly=3):
    """Smooth noisy signal while handling NaNs."""
    x = np.array(x, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan_to_num(x)
    nans = np.isnan(x)
    if nans.any():
        not_nan = ~nans
        xp, fp = np.flatnonzero(not_nan), x[not_nan]
        x[nans] = np.interp(np.flatnonzero(nans), xp, fp)
    if window >= n:
        window = n-1 if (n % 2 == 0) else n
    window = max(5, window if window % 2 == 1 else window-1)
    return savgol_filter(x, window_length=window, polyorder=min(poly, 3))


def compute_wgs84_derived_speed(df):
    """
    Compute speed from WGS84 GPS position data using numerical differentiation.

    Method:
    1. Convert WGS84 (lat, lon) to UTM coordinates (meters)
    2. Compute velocity using Savitzky-Golay filter differentiation
    3. Return speed magnitude in m/s

    This provides an independent ground truth for comparison with GPS-reported speed.
    """
    if 'locationLatitude(WGS84)' not in df.columns or 'locationLongitude(WGS84)' not in df.columns:
        print("⚠️  WGS84 coordinates not found, skipping derived speed calculation")
        return None

    # Remove rows with NaN coordinates and timestamps
    valid_mask = ~(df['locationLatitude(WGS84)'].isna() |
                   df['locationLongitude(WGS84)'].isna() |
                   df['locationTimestamp_since1970(s)'].isna())

    if valid_mask.sum() < 10:
        print("⚠️  Insufficient valid GPS coordinates for speed derivation")
        return None

    df_valid = df[valid_mask].copy()

    # Remove duplicate timestamps (keep first occurrence)
    df_valid = df_valid.drop_duplicates(subset=['locationTimestamp_since1970(s)'], keep='first')

    # Sort by timestamp
    df_valid = df_valid.sort_values('locationTimestamp_since1970(s)').reset_index(drop=True)

    if len(df_valid) < 10:
        print("⚠️  Insufficient unique GPS samples for speed derivation")
        return None

    # Determine UTM zone from first coordinate
    first_lon = df_valid['locationLongitude(WGS84)'].iloc[0]
    first_lat = df_valid['locationLatitude(WGS84)'].iloc[0]
    utm_zone = int((first_lon + 180) / 6) + 1
    is_northern = first_lat >= 0

    # Transform to UTM
    transformer = Transformer.from_crs(
        crs_from="epsg:4326",  # WGS84
        crs_to=f"epsg:{32600 + utm_zone if is_northern else 32700 + utm_zone}",
        always_xy=True
    )

    utm_x, utm_y = transformer.transform(
        df_valid['locationLongitude(WGS84)'].values,
        df_valid['locationLatitude(WGS84)'].values
    )

    # Get timestamps and compute dt
    t = df_valid['locationTimestamp_since1970(s)'].values
    t_rel = t - t[0]

    # Check for valid time differences
    time_diffs = np.diff(t_rel)
    if np.any(time_diffs <= 0) or np.median(time_diffs) <= 0:
        print("⚠️  Invalid timestamp sequence, skipping WGS84 speed derivation")
        return None

    # Compute velocity using Savitzky-Golay differentiation
    window = min(51, len(utm_x))
    if window % 2 == 0:
        window -= 1
    window = max(5, window)

    dt = np.median(time_diffs)

    # Ensure dt is valid and not too small
    if dt <= 0 or np.isnan(dt) or np.isinf(dt):
        print(f"⚠️  Invalid time delta: {dt}, skipping WGS84 speed derivation")
        return None

    try:
        vx = savgol_filter(utm_x, window_length=window, polyorder=3, deriv=1, delta=dt)
        vy = savgol_filter(utm_y, window_length=window, polyorder=3, deriv=1, delta=dt)
    except Exception as e:
        print(f"⚠️  Error computing velocity derivatives: {e}")
        return None

    # Speed magnitude
    v_mag = np.sqrt(vx**2 + vy**2)

    # Create result dataframe aligned with original
    result = pd.DataFrame({
        'locationTimestamp_since1970(s)': t,
        'speed_wgs84_derived(m/s)': v_mag
    })

    return result


# ============================================================
# ----------------------- Plotting ---------------------------
# ============================================================

def plot_acc_and_speed(df_orig, df_car, df_wgs84_speed, out_path, use_imperial_units=False):
    """Plot vehicle speed and acceleration with GPS vs WGS84-derived speed comparison."""

    # --- Conversion constants ---
    MS_TO_MPH = 2.23694     # 1 m/s = 2.23694 mph
    MS2_TO_MPH_PER_S = 2.23694  # 1 m/s² = 2.23694 mph/s (same conversion)

    # --- Unit labels ---
    speed_label = "Speed (m/s)"
    acc_label = "Acceleration (m/s²)"
    if use_imperial_units:
        speed_label = "Speed (mph)"
        acc_label = "Acceleration (mph/s)"

    # --- GPS speed ---
    if "locationSpeed(m/s)" not in df_orig.columns:
        print(f"⚠️  No GPS speed found in {out_path}")
        return

    speed_gps = df_orig["locationSpeed(m/s)"].to_numpy(dtype=float)
    t_loc = df_orig["locationTimestamp_since1970(s)"].to_numpy(dtype=float)
    t_loc_rel = t_loc - np.nanmin(t_loc)

    # --- Convert speed units ---
    if use_imperial_units:
        speed_gps *= MS_TO_MPH

    # --- WGS84-derived speed ---
    has_wgs84 = df_wgs84_speed is not None
    if has_wgs84:
        # Merge WGS84 speed with original dataframe
        df_merged = df_orig.merge(df_wgs84_speed, on='locationTimestamp_since1970(s)', how='left')
        speed_wgs84 = df_merged["speed_wgs84_derived(m/s)"].to_numpy(dtype=float)
        if use_imperial_units:
            speed_wgs84 *= MS_TO_MPH

    # --- Car-frame acceleration ---
    t_acc = df_car["motionTimestamp_sinceReboot(s)"].to_numpy(dtype=float)
    t_acc_rel = t_acc - np.nanmin(t_acc)

    acc_x = df_car["acc_car_x(ms2)"].to_numpy(dtype=float)
    acc_mag = df_car["acc_car_mag(ms2)"].to_numpy(dtype=float)

    # --- Convert acceleration units ---
    if use_imperial_units:
        acc_x *= MS2_TO_MPH_PER_S
        acc_mag *= MS2_TO_MPH_PER_S

    # --- Smooth for better visualization ---
    speed_gps_s = smooth_signal(speed_gps, window=101)
    if has_wgs84:
        speed_wgs84_s = smooth_signal(speed_wgs84, window=101)
    acc_x_s = smooth_signal(acc_x, window=51)
    acc_mag_s = smooth_signal(acc_mag, window=51)

    # --- Compute validation metrics ---
    if has_wgs84:
        valid_mask = ~(np.isnan(speed_gps) | np.isnan(speed_wgs84))
        if valid_mask.sum() > 0:
            rmse = np.sqrt(np.mean((speed_gps[valid_mask] - speed_wgs84[valid_mask])**2))
            correlation = np.corrcoef(speed_gps[valid_mask], speed_wgs84[valid_mask])[0, 1]
            print(f"📊 Speed Validation Metrics:")
            print(f"   RMSE: {rmse:.4f} {speed_label.split('(')[1].split(')')[0]}")
            print(f"   Correlation: {correlation:.4f}")

    # --- Plot ---
    fig, axs = plt.subplots(3 if has_wgs84 else 2, 1, figsize=(14, 12 if has_wgs84 else 8), sharex=False)

    # Speed comparison plot
    axs[0].plot(t_loc_rel, speed_gps, label="GPS-reported (raw)", linewidth=0.8, alpha=0.5, color='blue')
    axs[0].plot(t_loc_rel, speed_gps_s, label="GPS-reported (smoothed)", linewidth=1.4, color='darkblue')
    if has_wgs84:
        axs[0].plot(t_loc_rel, speed_wgs84, label="WGS84-derived (raw)", linewidth=0.8, alpha=0.5, color='orange')
        axs[0].plot(t_loc_rel, speed_wgs84_s, label="WGS84-derived (smoothed)", linewidth=1.4, color='darkorange')
    axs[0].set_ylabel(speed_label)
    axs[0].set_title("Vehicle Speed: GPS-reported vs WGS84-derived Ground Truth")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.4)

    # Speed difference plot (if WGS84 available)
    if has_wgs84:
        speed_diff = speed_gps - speed_wgs84
        speed_diff_s = smooth_signal(speed_diff, window=101)
        axs[1].plot(t_loc_rel, speed_diff, label="Difference (raw)", linewidth=0.6, alpha=0.5, color='red')
        axs[1].plot(t_loc_rel, speed_diff_s, label="Difference (smoothed)", linewidth=1.2, color='darkred')
        axs[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        axs[1].set_ylabel(f"Speed Difference {speed_label.split('(')[1]}")
        axs[1].set_title("Speed Difference: GPS-reported - WGS84-derived")
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.4)

    # Acceleration plot
    acc_idx = 2 if has_wgs84 else 1
    axs[acc_idx].plot(t_acc_rel, acc_x, label="Forward (X) raw", linewidth=0.6, alpha=0.7)
    axs[acc_idx].plot(t_acc_rel, acc_x_s, label="Forward (X) smoothed", linewidth=1.2)
    axs[acc_idx].plot(t_acc_rel, acc_mag_s, label="Total |a|", linestyle="--", alpha=0.8)
    axs[acc_idx].set_xlabel("Time (s) — relative to each sensor start")
    axs[acc_idx].set_ylabel(acc_label)
    axs[acc_idx].set_title("Car-frame Acceleration (from Smartphone Quaternion + Heading)")
    axs[acc_idx].legend()
    axs[acc_idx].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Plot saved: {out_path}")


# ============================================================
# ------------------ Batch Processing Loop -------------------
# ============================================================

def process_all_files(root_folder, save_csv=True, use_imperial_units=False):
    csv_files = [f for f in os.listdir(root_folder) if f.endswith(".csv") and not f.endswith("_motion_variables.csv")]
    print(f"📁 Found {len(csv_files)} CSV files under {root_folder}")

    csv_files = [csv_files[0]]

    for file in csv_files:
        pdb.set_trace()
        file_path = os.path.join(root_folder, file)
        try:
            df = pd.read_csv(file_path)
            print(f"\n{'='*60}")
            print(f"Processing {file} ({len(df)} rows) ...")
            print(f"{'='*60}")

            # --- Compute car-frame acceleration ---
            df_car = compute_car_acceleration(df)

            # --- Compute WGS84-derived speed for ground truth comparison ---
            df_wgs84_speed = compute_wgs84_derived_speed(df)

            # --- Save motion variables CSV ---
            if save_csv:
                out_csv = file_path.replace(".csv", "_motion_variables.csv")
                df_car.to_csv(out_csv, index=False)
                print(f"💾 Saved car-frame accelerations: {out_csv}")

                # Save WGS84-derived speed if available
                if df_wgs84_speed is not None:
                    out_wgs84_csv = file_path.replace(".csv", "_wgs84_speed.csv")
                    df_wgs84_speed.to_csv(out_wgs84_csv, index=False)
                    print(f"💾 Saved WGS84-derived speed: {out_wgs84_csv}")

            # --- Generate plots ---
            out_png = file_path.replace(".csv", "_motion_variables.png")
            plot_acc_and_speed(df, df_car, df_wgs84_speed, out_png, use_imperial_units=use_imperial_units)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print("\n🏁 All files processed.")


# ============================================================
# ------------------------- Main -----------------------------
# ============================================================

if __name__ == "__main__":
    root_folder = "/home/ubuntu/atai_USAA_DrivingBehaviourAnalysis_project/datasets/atai/data/driving_data/EQE_wireless_charging_port/"

    # Set this flag to True for mph & mph/s instead of m/s & m/s²
    use_imperial_units = True

    process_all_files(root_folder, save_csv=True, use_imperial_units=use_imperial_units)
