from typing import Tuple

import numpy as np


def stabcull(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Estimates the parameters of a stable distribution using McCulloch's method.

    Parameters:
        x (np.ndarray): Input data array.

    Returns:
        Tuple[float, float, float, float]: A tuple containing:
            - alpha (float): Stability parameter (1 < alpha <= 2).
            - sigma (float): Scale parameter (sigma > 0).
            - beta (float): Skewness parameter (-1 <= beta <= 1).
            - mu (float): Location parameter.

    References:
        [1] I.A.Koutrouvelis (1980) "Regression-Type Estimation of the
        Parameters of Stable Laws", JASA 75, 918-928.
        [2] R.Weron (1995) "Performance of the estimators of stable law
        parameters".

    Notes:
        This implementation is based on Matlab code provided by:
        M.Banys, M.Lach, S.Niedziela, S.Szymanski, R.Weron, S.Borak.
    """

    # Sort the data and calculate percentiles
    x = np.sort(x)
    x05, x25, x50, x75, x95 = np.percentile(x, [5, 25, 50, 75, 95])

    # Calculate va, vb, and vs based on McCulloch's method
    va = (x95 - x05) / (x75 - x25)
    vb = (x95 + x05 - 2 * x50) / (x95 - x05)
    vs = x75 - x25

    # Lookup tables
    tva = np.array([2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0])
    tvb = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ta = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    tb = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    psi1 = np.array(
        [
            [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
            [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
            [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
            [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
            [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
            [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
            [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
            [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
            [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
            [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
            [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
            [0.896, 0.892, 0.887, 0.883, 0.855, 0.823, 0.769],
            [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
            [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.595],
            [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513],
        ]
    )
    psi2 = np.array(
        [
            [0.000, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
            [0.000, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
            [0.000, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
            [0.000, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
            [0.000, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
            [0.000, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
            [0.000, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
            [0.000, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
            [0.000, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
            [0.000, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
            [0.000, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
            [0.000, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
            [0.000, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
            [0.000, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
            [0.000, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274],
        ]
    )

    psi3 = np.array(
        [
            [1.908, 1.908, 1.908, 1.908, 1.908],
            [1.914, 1.915, 1.916, 1.918, 1.921],
            [1.921, 1.922, 1.927, 1.936, 1.947],
            [1.927, 1.930, 1.943, 1.961, 1.987],
            [1.933, 1.940, 1.962, 1.997, 2.043],
            [1.939, 1.952, 1.988, 2.045, 2.116],
            [1.946, 1.967, 2.022, 2.106, 2.211],
            [1.955, 1.984, 2.067, 2.188, 2.333],
            [1.965, 2.007, 2.125, 2.294, 2.491],
            [1.980, 2.040, 2.205, 2.435, 2.696],
            [2.000, 2.085, 2.311, 2.624, 2.973],
            [2.040, 2.149, 2.461, 2.886, 3.356],
            [2.098, 2.244, 2.676, 3.265, 3.912],
            [2.189, 2.392, 3.004, 3.844, 4.775],
            [2.337, 2.635, 3.542, 4.808, 6.247],
            [2.588, 3.073, 4.534, 6.636, 9.144],
        ]
    )

    psi4 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.017, -0.032, -0.049, -0.064],
            [0.0, -0.030, -0.061, -0.092, -0.123],
            [0.0, -0.043, -0.088, -0.132, -0.179],
            [0.0, -0.056, -0.111, -0.170, -0.232],
            [0.0, -0.066, -0.134, -0.206, -0.283],
            [0.0, -0.075, -0.154, -0.241, -0.335],
            [0.0, -0.084, -0.173, -0.276, -0.390],
            [0.0, -0.090, -0.192, -0.310, -0.447],
            [0.0, -0.095, -0.208, -0.346, -0.508],
            [0.0, -0.098, -0.223, -0.383, -0.576],
            [0.0, -0.099, -0.237, -0.424, -0.652],
            [0.0, -0.096, -0.250, -0.469, -0.742],
            [0.0, -0.089, -0.262, -0.520, -0.853],
            [0.0, -0.078, -0.272, -0.581, -0.997],
            [0.0, -0.061, -0.279, -0.659, -1.198],
        ]
    )

    # Compute estimates for each column in x (assuming a 1D array here)
    # Determine indices for va and vb lookup
    tvai1 = np.max([[0] + list(np.where(tva <= 5)[0])])
    tvai2 = np.min([[14] + list(np.where(tva >= va)[0])])
    tvbi1 = np.max([[0] + list(np.where(tvb <= abs(vb))[0])])
    tvbi2 = np.min([[6] + list(np.where(tvb >= abs(vb))[0])])

    dista = (va - tva[tvai1]) / (tva[tvai2] - tva[tvai1]) if (tva[tvai2] - tva[tvai1]) != 0 else 0
    distb = (np.abs(vb) - tvb[tvbi1]) / (tvb[tvbi2] - tvb[tvbi1]) if (tvb[tvbi2] - tvb[tvbi1]) != 0 else 0

    # Linear interpolation for alpha and beta
    psi1b1 = dista * psi1[tvai2, tvbi1] + (1 - dista) * psi1[tvai1, tvbi1]
    psi1b2 = dista * psi1[tvai2, tvbi2] + (1 - dista) * psi1[tvai1, tvbi2]
    alpha = distb * psi1b2 + (1 - distb) * psi1b1

    psi2b1 = dista * psi2[tvai2, tvbi1] + (1 - dista) * psi2[tvai1, tvbi1]
    psi2b2 = dista * psi2[tvai2, tvbi2] + (1 - dista) * psi2[tvai1, tvbi2]
    beta = np.sign(vb) * (distb * psi2b2 + (1 - distb) * psi2b1)

    # Further interpolation for sigma
    tai1 = np.max([[0] + list(np.where(ta >= alpha)[0])])
    tai2 = np.min([[15] + list(np.where(ta <= alpha)[0])])
    tbi1 = np.max([[0] + list(np.where(tb <= np.abs(beta))[0])])
    tbi2 = np.min([[4] + list(np.where(tb >= np.abs(beta))[0])])

    dista = (alpha - ta[tai1]) / (ta[tai2] - ta[tai1]) if (ta[tai2] - ta[tai1]) != 0 else 0
    distb = (np.abs(beta) - tb[tbi1]) / (tb[tbi2] - tb[tbi1]) if (tb[tbi2] - tb[tbi1]) != 0 else 0

    psi3b1 = dista * psi3[tai2, tbi1] + (1 - dista) * psi3[tai1, tbi1]
    psi3b2 = dista * psi3[tai2, tbi2] + (1 - dista) * psi3[tai1, tbi2]
    sigma = vs / (distb * psi3b2 + (1 - distb) * psi3b1)

    # Compute zeta and mu
    psi4b1 = dista * psi4[tai2, tbi1] + (1 - dista) * psi4[tai1, tbi1]
    psi4b2 = dista * psi4[tai2, tbi2] + (1 - dista) * psi4[tai1, tbi2]
    zeta = np.sign(beta) * sigma * (distb * psi4b2 + (1 - distb) * psi4b1) + x50

    if abs(alpha - 1) < 0.05:
        mu = zeta
    else:
        mu = zeta - beta * sigma * np.tan(0.5 * np.pi * alpha)

    # Apply bounds
    alpha = np.clip(alpha, 1e-10, 2)
    sigma = np.clip(sigma, 1e-10, None)
    beta = np.clip(beta, -1, 1)

    return alpha, sigma, beta, mu
