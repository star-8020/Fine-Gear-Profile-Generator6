import numpy as np

def reflect_y(XX, YY):
    """Reflects coordinates across the Y-axis."""
    # Reverses the order of points and negates the Y values
    # to create a symmetrical shape across the vertical axis.
    return XX[::-1], -YY[::-1]

def translate(Xtemp, Ytemp, X_0, Y_0):
    """Translates coordinates by a given offset (X_0, Y_0)."""
    return Xtemp + X_0, Ytemp + Y_0

def rotate(Xtemp, Ytemp, ANGLE):
    """Rotates coordinates around the origin by a given ANGLE in radians."""
    XX = np.cos(ANGLE) * Xtemp - np.sin(ANGLE) * Ytemp
    YY = np.sin(ANGLE) * Xtemp + np.cos(ANGLE) * Ytemp
    return XX, YY

def create_circular_pattern(X_tooth, Y_tooth, Z, P_ANGLE, ALIGN_ANGLE):
    """Creates a full gear by rotating a single tooth profile."""
    all_X = []
    all_Y = []

    # Apply initial alignment rotation to the first tooth
    X_rot, Y_rot = rotate(X_tooth, Y_tooth, ALIGN_ANGLE)

    for i in range(int(Z)):
        # Rotate the aligned tooth to its final position
        X_final, Y_final = rotate(X_rot, Y_rot, P_ANGLE * i)
        all_X.append(X_final)
        all_Y.append(Y_final)

    return all_X, all_Y