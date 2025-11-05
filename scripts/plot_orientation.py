import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# ----------------------------
# Quaternion utilities
# ----------------------------
def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion (w, x, y, z) to a 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

# ----------------------------
# Main animation function
# ----------------------------
def animate_device_orientation(df,
                               quat_cols=['motionQuaternionW(R)', 'motionQuaternionX(R)',
                                          'motionQuaternionY(R)', 'motionQuaternionZ(R)'],
                               save_path="orientation_video.mp4",
                               fps=30,
                               dpi=100,
                               axis_length=1.0):
    """
    Animate and save the device orientation using quaternion data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing quaternion columns.
    quat_cols : list of str
        Column names for quaternion [w, x, y, z].
    save_path : str
        Output path for the MP4 file.
    fps : int
        Frames per second of the output video.
    dpi : int
        Resolution for saved frames.
    axis_length : float
        Length of the 3D axes to display.
    """

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set limits and labels (include units)
    lim = axis_length * 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel('X-axis (normalized units)')
    ax.set_ylabel('Y-axis (normalized units)')
    ax.set_zlabel('Z-axis (normalized units)')
    ax.set_title('Device Orientation')
    plt.tight_layout()

    origin = np.array([0, 0, 0])
    base_axes = np.eye(3)

    # Writer setup
    writer = FFMpegWriter(fps=fps)
    frames = len(df)

    # Initialize label placeholders
    text_labels = []

    print(f"🎬 Rendering {frames} frames at {fps} FPS...")
    with writer.saving(fig, save_path, dpi=dpi):
        for i in tqdm(range(frames), desc="Rendering video", unit="frame"):
            # Clear previous frame’s arrows and labels
            ax.collections.clear()
            for txt in text_labels:
                txt.remove()
            text_labels = []

            # Get quaternion and convert to rotation matrix
            w, x, y, z = df.loc[i, quat_cols]
            R = quaternion_to_rotation_matrix(w, x, y, z)
            rotated_axes = R @ base_axes

            # Draw rotated axes
            colors = ['r', 'g', 'b']
            labels = ['X', 'Y', 'Z']

            for j in range(3):
                axis_vector = rotated_axes[:, j] * axis_length
                ax.quiver(*origin, *axis_vector, color=colors[j], length=axis_length, normalize=True)

                # Add letter label at the end of each arrow
                text_pos = axis_vector * 1.05  # slightly beyond tip
                txt = ax.text(*text_pos, labels[j],
                              color=colors[j], fontsize=12, fontweight='bold')
                text_labels.append(txt)

            # Capture this frame
            writer.grab_frame()

    plt.close(fig)
    print(f"✅ Animation saved to: {save_path}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    root_folder = "/home/ubuntu/data/driving_data/EQE_wireless_charging_port/"
    file_path = os.path.join(root_folder, "2025-08-29_17_28_36_venrock_home_abrupt_brake.csv")

    df = pd.read_csv(file_path)

    first_n_seconds = 100
    average_frequency = 106
    n_samples = int(first_n_seconds * average_frequency)
    df_ = df.iloc[:n_samples].copy()

    animate_device_orientation(df_, save_path="device_orientation.mp4", fps=30)
