#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import math
import quaternion
import numpy as np

from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
    quaternion_from_coeff,
    quaternion_to_list,
)


def quaternion_xyzw_to_wxyz(v: np.array):
    return np.quaternion(v[3], *v[0:3])


def quaternion_wxyz_to_xyzw(v: np.array):
    return np.quaternion(*v[1:4], v[0])


def quaternion_to_coeff(quat: np.quaternion) -> np.array:
    r"""Converts a quaternions to coeffs in [x, y, z, w] format
    """
    coeffs = np.zeros((4,))
    coeffs[3] = quat.real
    coeffs[0:3] = quat.imag
    return coeffs


def compute_heading_from_quaternion(r):
    """
    r - rotation quaternion

    Computes clockwise rotation about Y.
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(r.inverse(), direction_vector)

    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()
    return phi


def compute_quaternion_from_heading(theta):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    theta - heading angle in radians --- measured clockwise from -Z to X.

    Compute quaternion that represents the corresponding clockwise rotation about Y axis.
    """
    # Real part
    q0 = math.cos(-theta / 2)
    # Imaginary part
    q = (0, math.sin(-theta / 2), 0)

    return np.quaternion(q0, *q)


def compute_egocentric_delta(p1, r1, p2, r2):
    """
    p1, p2 - (x, y, z) position
    r1, r2 - np.quaternions

    Compute egocentric change from (p1, r1) to (p2, r2) in
    the coordinates of (p1, r1)

    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    theta_1 = compute_heading_from_quaternion(r1)
    theta_2 = compute_heading_from_quaternion(r2)

    D_rho = math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    D_phi = (
        math.atan2(x2 - x1, -z2 + z1) - theta_1
    )  # counter-clockwise rotation about Y from -Z to X
    D_theta = theta_2 - theta_1

    return (D_rho, D_phi, D_theta)


def compute_updated_pose(p, r, delta_xz, delta_y):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.

    p - (x, y, z) position
    r - np.quaternion
    delta_xz - (D_rho, D_phi, D_theta) in egocentric coordinates
    delta_y - scalar change in height

    Compute new position after a motion of delta from (p, r)
    """
    x, y, z = p
    theta = compute_heading_from_quaternion(
        r
    )  # counter-clockwise rotation about Y from -Z to X
    D_rho, D_phi, D_theta = delta_xz

    xp = x + D_rho * math.sin(theta + D_phi)
    yp = y + delta_y
    zp = z - D_rho * math.cos(theta + D_phi)
    pp = np.array([xp, yp, zp])

    thetap = theta + D_theta
    rp = compute_quaternion_from_heading(thetap)

    return pp, rp
