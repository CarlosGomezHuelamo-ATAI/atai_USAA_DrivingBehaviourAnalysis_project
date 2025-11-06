# Car Acceleration Estimation Method

## Overview

This document describes the mathematical method used to estimate car-frame acceleration from smartphone sensor data in `scripts/compute_velocity_acceleration.py`.

## Problem Statement

Given:
- Smartphone accelerometer readings in the device coordinate frame
- Device orientation (quaternion)
- Vehicle heading (compass/GPS)

Goal:
- Compute acceleration in the car coordinate frame (forward, lateral, vertical)

## Coordinate Frame Definitions

### 1. Device Frame (Smartphone)
- **x**: Typically points right (landscape) or up (portrait)
- **y**: Typically points up (landscape) or left (portrait)
- **z**: Perpendicular to screen (outward)
- Frame depends on device mounting and orientation

### 2. Earth Frame (World/Inertial)
- **x**: East
- **y**: North
- **z**: Up (opposite to gravity)
- Fixed inertial reference frame

### 3. Car Frame (Vehicle)
- **x**: Forward (longitudinal direction)
- **y**: Left (lateral direction)
- **z**: Up (vertical direction)
- Aligned with vehicle motion

## Mathematical Formulation

### Step 1: Remove Gravity from Device Acceleration

The smartphone reports user acceleration with gravity already removed:

```
a_device = [ax_device, ay_device, az_device]  (units: G)
```

Convert from G to m/s²:

```
a_device_ms2 = a_device × 9.80665
```

**Implementation:** `scripts/compute_velocity_acceleration.py:52-56`

### Step 2: Quaternion to Rotation Matrix

The device orientation is given as a quaternion `q = (x, y, z, w)`. Convert to rotation matrix:

**Normalization:**
```
n = √(x² + y² + z² + w²)
q_norm = (x/n, y/n, z/n, w/n)
```

**Rotation Matrix (Device → Earth):**

```
R = | 1 - 2(y² + z²)     2(xy - zw)        2(xz + yw)      |
    | 2(xy + zw)         1 - 2(x² + z²)    2(yz - xw)      |
    | 2(xz - yw)         2(yz + xw)        1 - 2(x² + y²)  |
```

This rotation matrix transforms vectors from the device coordinate frame to the earth coordinate frame.

**Implementation:** `scripts/compute_velocity_acceleration.py:12-24`

### Step 3: Transform Device Acceleration to Earth Frame

Apply the rotation matrix to device acceleration:

```
a_earth = R · a_device_ms2
```

Result:
```
a_earth = [a_east, a_north, a_up]  (units: m/s²)
```

**Implementation:** `scripts/compute_velocity_acceleration.py:27-29, 63`

### Step 4: Transform Earth Frame to Car Frame

The vehicle heading θ (in degrees, clockwise from North) defines the car's orientation in the horizontal plane.

**Rotation about vertical (z) axis:**

Convert heading to radians:
```
θ_rad = θ × (π/180)
```

Rotation matrix (Earth → Car):
```
R_z(θ) = | cos(-θ)   -sin(-θ)   0 |
         | sin(-θ)    cos(-θ)   0 |
         | 0          0         1 |
```

Note: We use `-θ` because heading is measured clockwise from North, but standard rotation matrices use counter-clockwise rotations.

Apply rotation:
```
a_car = R_z(θ) · a_earth
```

Result:
```
a_car = [a_forward, a_lateral, a_vertical]  (units: m/s²)
```

**Implementation:** `scripts/compute_velocity_acceleration.py:32-39, 71`

### Step 5: Compute Acceleration Magnitude

Total acceleration magnitude:
```
|a_car| = √(a_forward² + a_lateral² + a_vertical²)
```

**Implementation:** `scripts/compute_velocity_acceleration.py:78`

## Complete Transformation Pipeline

Combining all steps:

```
a_car = R_z(-θ) · R_quaternion(qx, qy, qz, qw) · a_device · g
```

Where:
- `a_device`: User acceleration from smartphone (G units)
- `g = 9.80665 m/s²`: Gravitational constant
- `R_quaternion`: Rotation matrix from quaternion
- `R_z(-θ)`: Rotation about vertical axis by heading angle
- `a_car`: Final car-frame acceleration (m/s²)

## Heading Source Selection

The algorithm uses a priority-based heading selection:

1. **Primary:** `locationTrueHeading(°)` - GPS-derived true heading
2. **Fallback:** `motionHeading(°)` - Magnetometer-derived heading

```python
heading = locationTrueHeading if available else motionHeading
```

GPS heading is preferred because it's more accurate at higher speeds and not affected by magnetic interference.

**Implementation:** `scripts/compute_velocity_acceleration.py:65-69`

## Output Variables

The function `compute_car_acceleration()` returns a DataFrame with:

| Column | Description | Units |
|--------|-------------|-------|
| `motionTimestamp_sinceReboot(s)` | Timestamp of measurement | seconds |
| `acc_car_x(ms2)` | Forward acceleration | m/s² |
| `acc_car_y(ms2)` | Lateral acceleration | m/s² |
| `acc_car_z(ms2)` | Vertical acceleration | m/s² |
| `acc_car_mag(ms2)` | Total acceleration magnitude | m/s² |

## Assumptions and Limitations

### Assumptions:
1. Device orientation (quaternion) is accurately computed by the smartphone OS
2. Heading information (GPS or magnetometer) is available and accurate
3. User acceleration has gravity already removed by the device
4. The car is moving on a relatively flat surface (pitch/roll not explicitly modeled beyond quaternion)

### Limitations:
1. **Heading accuracy**:
   - GPS heading is inaccurate at low speeds or when stationary
   - Magnetometer heading can be affected by magnetic interference

2. **Quaternion accuracy**:
   - Device orientation estimation may drift over time
   - Accuracy depends on smartphone sensor fusion algorithm

3. **Mounting variations**:
   - Method assumes device can maintain orientation tracking
   - Works best with securely mounted devices

4. **Sensor noise**:
   - Raw accelerometer data contains noise
   - Smoothing (Savitzky-Golay filter) is applied for visualization only

## Validation

### Speed Comparison (Ground Truth)

Two independent speed measurements are compared:

1. **GPS-reported speed**: Direct speed measurement from GPS (`locationSpeed(m/s)`)
2. **WGS84-derived speed**: Speed computed from position changes using numerical differentiation

**WGS84-Derived Speed Method:**
- Convert GPS coordinates (lat, lon) to UTM (meters)
- Apply Savitzky-Golay filter for smooth differentiation
- Compute velocity: `v = √(vx² + vy²)`

**Validation metrics:**
- RMSE (Root Mean Square Error)
- Correlation coefficient

These metrics quantify the agreement between GPS-reported and position-derived speeds, validating the GPS speed measurements.

**Implementation:** `scripts/compute_velocity_acceleration.py:106-199, 233-241`

## References

1. **Quaternion rotation formulas**: Standard quaternion-to-rotation-matrix conversion
2. **Coordinate transformations**: Earth-Centered Earth-Fixed (ECEF) and local tangent plane
3. **Savitzky-Golay filter**: For smooth numerical differentiation
4. **UTM projection**: Universal Transverse Mercator for metric coordinate system

## Code Location

Main implementation: `scripts/compute_velocity_acceleration.py`

Key functions:
- `quat_to_rot_matrix()`: Lines 12-24
- `rotate_device_to_earth()`: Lines 27-29
- `earth_to_car()`: Lines 32-39
- `compute_car_acceleration()`: Lines 46-81
- `compute_wgs84_derived_speed()`: Lines 106-199

## Usage Example

```python
import pandas as pd
from compute_velocity_acceleration import compute_car_acceleration

# Load sensor data
df = pd.read_csv("driving_session.csv")

# Compute car-frame acceleration
df_car_acc = compute_car_acceleration(df)

# Access results
forward_acc = df_car_acc["acc_car_x(ms2)"]
lateral_acc = df_car_acc["acc_car_y(ms2)"]
```
