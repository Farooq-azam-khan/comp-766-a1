from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from convolution import ORIENTATION_COUNT

type Position = Sequence[float] | NDArray[np.number]


def gamma(beta: float, gamma: float) -> float:
    if not 0 <= beta <= np.pi:
        raise ValueError(f'{beta=} not in range 0 to pi.')

    if not 0 <= gamma <= np.pi:
        raise ValueError(f'{gamma=} not in range 0 to pi.')

    difference = gamma - beta
    if difference > np.pi / 2:
        return difference - np.pi
    if difference < -np.pi / 2:
        return difference + np.pi
    return difference


def is_cocircular(
    posi: Position,
    posj: Position,
    theta_lam: float,
    theta_lam_prime: float,
) -> bool:
    epsilon = np.pi / ORIENTATION_COUNT
    dx = float(posj[0]) - float(posi[0])
    dy = float(posj[1]) - float(posi[1])
    d_ij = np.hypot(dx, dy)
    if not np.isfinite(d_ij) or d_ij < 1:
        raise ValueError('Positions must be finite and at least one pixel apart.')

    alpha = np.arcsin(1 / d_ij)
    theta_ij = float(np.arctan2(dy, dx) % np.pi)
    return bool(
        np.abs(gamma(theta_lam, theta_ij) - gamma(theta_ij, theta_lam_prime))
        < epsilon + 2 * alpha
    )


def cocircularity_coeff(
    posi: Position,
    posj: Position,
    lam: int,
    lam_prime: int,
    c_min: float = 0.1,
) -> float:
    theta_lam = lam * np.pi / ORIENTATION_COUNT
    theta_lam_prime = lam_prime * np.pi / ORIENTATION_COUNT
    if is_cocircular(posi, posj, theta_lam, theta_lam_prime):
        return 1.0
    # fig 9: not doing too much geometry
    # drop_off_slope = 0.1 # eta in the paper.
    return c_min


def generate_support() -> None:
    pass
