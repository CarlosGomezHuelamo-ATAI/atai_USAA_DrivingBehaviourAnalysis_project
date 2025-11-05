# Author: Carlos Gómez Huélamo
# Company: Archetype AI
# Date: October 22nd 2025

# We can compute a rotation matrix from the sensor frame (typically device axes) to the car frame using accelerometer and magnetometer readings without relying on Android’s built-in functions. I’ll give a clean Python implementation.

# Steps:

# Accelerometer → gives the gravity vector (down direction).
# Magnetometer → gives the geomagnetic vector (rough North).
# From these two, we can compute three orthogonal axes for the rotation matrix.

def compute_rotation_matrix(accel, magnet):
    """
    Compute rotation matrix from sensor frame to car frame.

    Parameters:
        accel: np.array of shape (3,), accelerometer readings [ax, ay, az]
        magnet: np.array of shape (3,), magnetometer readings [mx, my, mz]

    Returns:
        R: np.array of shape (3,3), rotation matrix
           such that v_car = R @ v_sensor
    """
    # Normalize accelerometer (gravity)
    g = accel / np.linalg.norm(accel)
    
    # Normalize magnetometer
    m = magnet / np.linalg.norm(magnet)
    
    # East = cross product of magnet and gravity
    east = np.cross(m, g)
    east /= np.linalg.norm(east)
    
    # North = cross product of gravity and east
    north = np.cross(g, east)
    
    # Rotation matrix: columns are [east, north, -gravity]
    R = np.column_stack((east, north, -g))
    
    return R