# Author: Carlos Gómez Huélamo
# Company: Archetype AI
# Date: October 22nd 2025

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import numpy as np

from matplotlib.patches import Rectangle

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./data"  # Path to your dataset
SAVE_FIGS = True     # Enable saving
FIGURES_DIR = "figures"  # Save output images here
PLOT = True

EVENT_INFO = {
    "evento_nao_agressivo": {
        "color": "yellow",
        "label_en": "Non-aggressive event"
    },
    "curva_direita_agressiva": {
        "color": "red",
        "label_en": "Aggressive right turn"
    },
    "curva_esquerda_agressiva": {
        "color": "blue",
        "label_en": "Aggressive left turn"
    },
    "troca_faixa_direita_agressiva": {
        "color": "orange",
        "label_en": "Aggressive right lane change"
    },
    "troca_faixa_esquerda_agressiva": {
        "color": "green",
        "label_en": "Aggressive left lane change"
    },
    "freada_agressiva": {
        "color": "purple",
        "label_en": "Aggressive braking"
    }
}

# ============================================================
# DATA HELPERS
# ============================================================

def preprocess_sensor(df):
    """Convert uptimeNanos to time (s) starting from zero."""
    df = df.copy()
    if 'uptimeNanos' in df.columns:
        df['time'] = (df['uptimeNanos'] - df['uptimeNanos'].iloc[0]) / 1e9
    return df

def load_session(sess_path):
    """Load all sensor data and metadata for one session."""
    data = {}
    try:
        data['accel_raw'] = preprocess_sensor(pd.read_csv(os.path.join(sess_path, 'acelerometro_terra.csv')))
        data['accel_linear'] = preprocess_sensor(pd.read_csv(os.path.join(sess_path, 'aceleracaoLinear_terra.csv')))
        data['gyro'] = preprocess_sensor(pd.read_csv(os.path.join(sess_path, 'giroscopio_terra.csv')))
        data['mag'] = preprocess_sensor(pd.read_csv(os.path.join(sess_path, 'campoMagnetico_terra.csv')))

        # Load and clean ground truth
        gt_path = os.path.join(sess_path, 'groundTruth.csv')
        gt = pd.read_csv(gt_path)
        gt.columns = [c.strip().lower().replace(' ', '') for c in gt.columns]  # Clean column names

        # Add English translation column
        if 'evento' in gt.columns:
            gt['evento_en'] = gt['evento'].map(lambda e: EVENT_INFO.get(e, {}).get('label_en', e))

        data['ground_truth'] = gt

        # Metadata
        meta_path = os.path.join(sess_path, 'viagem.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data['meta'] = json.load(f)

    except FileNotFoundError as e:
        print(f"Missing file in {sess_path}: {e}")
        return None

    return data

def find_sessions(base_dir):
    """Find available session folders."""
    return [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

# ============================================================
# PLOTTING
# ============================================================

def plot_sensor_with_events(df, events, session_name, title, save=False):
    """Plot x,y,z sensor data with colored segments for each event."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    axes = axes.ravel()
    components = ['x', 'y', 'z']

    for i, comp in enumerate(components):
        # Plot normal signal thinner and lighter
        axes[i].plot(df['time'], df[comp], color='gray', linewidth=0.8)
        axes[i].set_ylabel(f'{comp} axis')
        axes[i].grid(True, alpha=0.3)

        # Overlay maneuvers with thicker line
        for _, row in events.iterrows():
            event = row['evento']
            start, end = row['inicio'], row['fim']
            color = EVENT_INFO.get(event, {}).get('color', 'gray')

            # Select the data within the event time window
            mask = (df['time'] >= start) & (df['time'] <= end)
            axes[i].plot(df['time'][mask], df[comp][mask], color=color, linewidth=2.5)

        # --- Add dashed threshold lines for X-axis only ---
        if ((title == "Linear Acceleration" or title == "Raw Acceleration") and
        comp.lower() == 'x'):

            def get_segments(mask):
                """Return start/end indices for contiguous True regions."""
                mask = mask.astype(int)
                diff = mask.diff().fillna(0)
                starts = df.index[diff == 1].tolist()
                ends = df.index[diff == -1].tolist()

                if mask.iloc[0] == 1:
                    starts.insert(0, df.index[0])
                if mask.iloc[-1] == 1:
                    ends.append(df.index[-1])

                return list(zip(starts, ends))

            def merge_segments(segments, gap=0.0):
                """Merge overlapping or close segments. 'gap' allows merging segments close in time."""
                if not segments:
                    return []

                segments = sorted(segments, key=lambda x: x[0])
                merged = [segments[0]]

                for start, end in segments[1:]:
                    last_start, last_end = merged[-1]
                    if start <= last_end + gap:  # overlapping or close
                        merged[-1] = (last_start, max(last_end, end))
                    else:
                        merged.append((start, end))
                return merged

            def exclude_event_overlap(exceed_segments, events):
                """Remove segments that overlap with any labeled events."""
                filtered = []
                for start, end in exceed_segments:
                    overlap = False
                    for _, row in events.iterrows():
                        ev_start, ev_end = row['inicio'], row['fim']
                        if not (end < ev_start or start > ev_end):
                            overlap = True
                            break
                    if not overlap:
                        filtered.append((start, end))
                return filtered

            # Thresholds for X-axis (USAA)
            accel_threshold = 2.593
            brake_threshold = -2.906
            ax = axes[i]

            # Threshold dashed lines
            ax.axhline(y=accel_threshold, color='red', linestyle='--', linewidth=1.5)
            ax.axhline(y=brake_threshold, color='blue', linestyle='--', linewidth=1.5)

            # Detect segments exceeding thresholds
            above_mask = df[comp] > accel_threshold
            below_mask = df[comp] < brake_threshold

            above_segments_idx = get_segments(above_mask)
            below_segments_idx = get_segments(below_mask)

            # Convert indices to time
            above_segments = [(df['time'].iloc[s], df['time'].iloc[e]) for s, e in above_segments_idx]
            below_segments = [(df['time'].iloc[s], df['time'].iloc[e]) for s, e in below_segments_idx]

            # Exclude segments overlapping with labeled events
            above_segments = exclude_event_overlap(above_segments, events)
            below_segments = exclude_event_overlap(below_segments, events)

            # Merge overlapping/close segments (NMS)
            gap = (df['time'].iloc[-1] - df['time'].iloc[0]) * 0.001  # 0.1% of total duration
            above_segments = merge_segments(above_segments, gap=gap)
            below_segments = merge_segments(below_segments, gap=gap)

            # Add small horizontal padding for visual clarity
            padding = (df['time'].iloc[-1] - df['time'].iloc[0]) * 0.005  # 0.5% of total duration

            # Draw bounding boxes
            for start_t, end_t in above_segments:
                start_t = max(df['time'].iloc[0], start_t - padding)
                end_t = min(df['time'].iloc[-1], end_t + padding)
                seg_mask = (df['time'] >= start_t) & (df['time'] <= end_t)
                ymin, ymax = accel_threshold, df[comp][seg_mask].max()

                ax.add_patch(Rectangle(
                    (start_t, ymin),
                    end_t - start_t,
                    ymax - ymin,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=2.5
                ))

            for start_t, end_t in below_segments:
                start_t = max(df['time'].iloc[0], start_t - padding)
                end_t = min(df['time'].iloc[-1], end_t + padding)
                seg_mask = (df['time'] >= start_t) & (df['time'] <= end_t)
                ymin, ymax = df[comp][seg_mask].min(), brake_threshold

                ax.add_patch(Rectangle(
                    (start_t, ymin),
                    end_t - start_t,
                    ymax - ymin,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=2.5
                ))

    axes[-1].set_xlabel("Time (s)")

    if title == "Linear Acceleration":
        title_aux = "Linear Acceleration [m/s2]"
    elif title == "Raw Acceleration":
        title_aux = "Raw Acceleration [m/s2]"
    elif title == "Gyroscope":
        title_aux = "Gyroscope [rad/s]"
    elif title == "Magnetic Field":
        title_aux = "Magnetic Field [μT]"

    fig.suptitle(f"{title_aux} - Session {session_name}", fontsize=14)

    # Build the legend (maneuvers + normal behavior)
    legend_patches = [
    plt.Line2D([0], [0], color='gray', lw=2, label='Normal behavior'),
    plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5,
            label='Rapid acceleration threshold USAA (+2.593)'),
    plt.Line2D([0], [0], color='blue', linestyle='--', lw=1.5,
            label='Harsh braking threshold USAA (-2.906)')
    ] + [
        plt.Line2D([0], [0], color=info["color"], lw=6, alpha=0.8, label=info["label_en"])
        for event, info in EVENT_INFO.items()
    ]
    fig.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(FIGURES_DIR, f"{session_name}_{title.replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

def main():
    sessions = find_sessions(DATA_DIR)
    print(f"Found sessions: {sessions}\n")
    # sessions = ["16"]  # Hardcoded for now

    for sess in sessions:
        print(f"=== Loading session {sess} ===")
        sess_path = os.path.join(DATA_DIR, sess)
        data = load_session(sess_path)
        if data is None:
            continue

        gt = data['ground_truth']

        if PLOT:
            plot_sensor_with_events(data['accel_linear'], gt, sess, 'Linear Acceleration', save=SAVE_FIGS)
            plot_sensor_with_events(data['accel_raw'], gt, sess, 'Raw Acceleration', save=SAVE_FIGS)
            plot_sensor_with_events(data['gyro'], gt, sess, 'Gyroscope', save=SAVE_FIGS)
            plot_sensor_with_events(data['mag'], gt, sess, 'Magnetic Field', save=SAVE_FIGS)


if __name__ == "__main__":
    main()
