import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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


# ============================================================
# ----------------------- Plotting ---------------------------
# ============================================================

def plot_acc_and_speed(df_orig, df_car, out_path, use_imperial_units=False):
    """Plot vehicle speed and acceleration (metric or imperial)."""
    
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

    speed = df_orig["locationSpeed(m/s)"].to_numpy(dtype=float)
    t_loc = df_orig["locationTimestamp_since1970(s)"].to_numpy(dtype=float)
    t_loc_rel = t_loc - np.nanmin(t_loc)

    # --- Convert speed units ---
    if use_imperial_units:
        speed *= MS_TO_MPH

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
    speed_s = smooth_signal(speed, window=101)
    acc_x_s = smooth_signal(acc_x, window=51)
    acc_mag_s = smooth_signal(acc_mag, window=51)

    # --- Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axs[0].plot(t_loc_rel, speed, label="Speed (raw)", linewidth=0.8, alpha=0.6)
    axs[0].plot(t_loc_rel, speed_s, label="Speed (smoothed)", linewidth=1.4)
    axs[0].set_ylabel(speed_label)
    axs[0].set_title("Vehicle Speed (GPS)")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.4)

    axs[1].plot(t_acc_rel, acc_x, label="Forward (X) raw", linewidth=0.6, alpha=0.7)
    axs[1].plot(t_acc_rel, acc_x_s, label="Forward (X) smoothed", linewidth=1.2)
    axs[1].plot(t_acc_rel, acc_mag_s, label="Total |a|", linestyle="--", alpha=0.8)
    axs[1].set_xlabel("Time (s) — relative to each sensor start")
    axs[1].set_ylabel(acc_label)
    axs[1].set_title("Car-frame Acceleration (from Smartphone Quaternion + Heading)")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Plot saved: {out_path}")


# ============================================================
# ------------------ Batch Processing Loop -------------------
# ============================================================

def process_all_files(root_folder, save_csv=True, use_imperial_units=False):
    csv_files = [f for f in os.listdir(root_folder) if f.endswith(".csv")]
    print(f"📁 Found {len(csv_files)} CSV files under {root_folder}")

    for file in csv_files:
        file_path = os.path.join(root_folder, file)
        try:
            df = pd.read_csv(file_path)
            print(f"Processing {file} ({len(df)} rows) ...")

            df_car = compute_car_acceleration(df)

            # --- Save motion variables CSV ---
            if save_csv:
                out_csv = file_path.replace(".csv", "_motion_variables.csv")
                df_car.to_csv(out_csv, index=False)
                print(f"💾 Saved car-frame accelerations: {out_csv}")

            # --- Generate plots ---
            out_png = file_path.replace(".csv", "_motion_variables.png")
            plot_acc_and_speed(df, df_car, out_png, use_imperial_units=use_imperial_units)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print("🏁 All files processed.")


# ============================================================
# ------------------------- Main -----------------------------
# ============================================================

if __name__ == "__main__":
    root_folder = "/home/ubuntu/data/driving_data/EQE_wireless_charging_port/"

    # Set this flag to True for mph & mph/s instead of m/s & m/s²
    use_imperial_units = True

    process_all_files(root_folder, save_csv=False, use_imperial_units=use_imperial_units)
