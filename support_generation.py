from collections.abc import Sequence
from dataclasses import dataclass

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


def default_curvature_edges(
    neighborhood_radius: int,
    max_curvature: float = 0.2,
    class_count: int = 7,
) -> NDArray[np.float64]:
    if neighborhood_radius < 1:
        raise ValueError('Neighborhood radius must be positive.')
    if not np.isfinite(max_curvature) or max_curvature <= 0:
        raise ValueError('Maximum curvature must be finite and positive.')
    if class_count < 1 or class_count % 2 == 0:
        raise ValueError('Use a positive odd number of curvature classes.')
    if class_count == 1:
        return np.array([-max_curvature, max_curvature])
    straight_limit = min(2 / neighborhood_radius ** 2, max_curvature / 2)
    positive_edges = np.linspace(straight_limit, max_curvature, class_count // 2 + 1)
    return np.concatenate((-positive_edges[::-1], positive_edges))


@dataclass(frozen=True)
class _SupportOffset:
    dy: int
    dx: int
    labels: NDArray[np.int64]
    target_classes: NDArray[np.int64]
    neighbor_classes: NDArray[np.int64]
    coefficients: NDArray[np.float64]


def _curvature_classes(
    curvatures: NDArray[np.float64], edges: NDArray[np.float64]
) -> NDArray[np.int64]:
    indices = np.searchsorted(edges, curvatures, side="right") - 1
    indices[curvatures == edges[-1]] = len(edges) - 2
    indices[(curvatures < edges[0]) | (curvatures > edges[-1])] = -1
    return indices.astype(np.int64)


def _support_offsets(
    angles: NDArray[np.float64],
    radius: int,
    edges: NDArray[np.float64],
    c_min: float,
) -> list[_SupportOffset]:
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance_squared = dx * dx + dy * dy
            if distance_squared == 0 or distance_squared > radius * radius:
                continue
            chord_angle = np.arctan2(dy, dx) % np.pi
            differences = chord_angle - angles
            interior_angles = np.where(
                differences > np.pi / 2,
                differences - np.pi,
                np.where(differences < -np.pi / 2, differences + np.pi, differences),
            )
            tolerance = np.pi / ORIENTATION_COUNT + 2 * np.arcsin(
                1 / np.sqrt(distance_squared)
            )
            coefficients = np.where(
                np.abs(interior_angles[:, None] + interior_angles[None, :]) < tolerance,
                1.0,
                c_min,
            )
            curvatures = 2 * (np.cos(angles) * dy - np.sin(angles) * dx) / distance_squared
            target_classes = _curvature_classes(curvatures, edges)
            neighbor_classes = _curvature_classes(-curvatures, edges)
            labels = np.flatnonzero(target_classes >= 0)
            if labels.size:
                offsets.append(_SupportOffset(
                    dy, dx, labels, target_classes[labels], neighbor_classes,
                    coefficients[labels],
                ))
    return offsets


def _accumulate_support(
    probabilities: NDArray[np.float64],
    offsets: list[_SupportOffset],
    class_count: int,
    curvature_classes: NDArray[np.int64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    label_count, height, width = probabilities.shape
    support = np.zeros_like(probabilities)
    winning_classes = np.full(probabilities.shape, -1, dtype=np.int64)
    tile_height = max(1, min(height, 8_000_000 // (class_count * label_count * width)))
    for row_start in range(0, height, tile_height):
        row_stop = min(height, row_start + tile_height)
        class_support = np.zeros(
            (class_count, label_count, row_stop - row_start, width), dtype=np.float64
        )
        for offset in offsets:
            y0, y1 = max(row_start, -offset.dy), min(row_stop, height - offset.dy)
            x0, x1 = max(0, -offset.dx), min(width, width - offset.dx)
            if y0 >= y1 or x0 >= x1:
                continue
            neighbor_y = slice(y0 + offset.dy, y1 + offset.dy)
            neighbor_x = slice(x0 + offset.dx, x1 + offset.dx)
            neighbors = probabilities[:, neighbor_y, neighbor_x]
            if curvature_classes is not None:
                consistent = (
                    curvature_classes[:, neighbor_y, neighbor_x]
                    == offset.neighbor_classes[:, None, None]
                ) & (offset.neighbor_classes[:, None, None] >= 0)
                neighbors = neighbors * consistent
            contribution = np.einsum(
                "ij,jyx->iyx", offset.coefficients, neighbors, optimize=True
            )
            class_support[
                offset.target_classes, offset.labels,
                y0 - row_start:y1 - row_start, x0:x1,
            ] += contribution
        best = class_support.max(axis=0)
        support[:, row_start:row_stop] = best
        winning_classes[:, row_start:row_stop] = np.where(
            best > 0, class_support.argmax(axis=0), -1
        )
    return support, winning_classes


def generate_support(
    probabilities: NDArray[np.float64],
    tangent_angles: NDArray[np.float64],
    neighborhood_radius: int,
    curvature_edges: NDArray[np.float64],
    *,
    curvature_classes: NDArray[np.int64] | None = None,
    c_min: float = 0.1,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if probabilities.ndim != 3 or any(size == 0 for size in probabilities.shape):
        raise ValueError('Probabilities must have shape (orientations, height, width).')
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError('Probabilities must be finite and nonnegative.')
    if tangent_angles.shape != (probabilities.shape[0],):
        raise ValueError('Provide one tangent angle per orientation.')
    if not np.all(np.isfinite(tangent_angles)) or np.any(
        (tangent_angles < 0) | (tangent_angles > np.pi)
    ):
        raise ValueError('Tangent angles must be finite and in range 0 to pi.')
    if not isinstance(neighborhood_radius, int) or neighborhood_radius < 1:
        raise ValueError('Neighborhood radius must be a positive integer.')
    if curvature_edges.ndim != 1 or curvature_edges.size < 2 or not np.all(
        np.isfinite(curvature_edges)
    ) or np.any(np.diff(curvature_edges) <= 0):
        raise ValueError('Curvature edges must be finite and strictly increasing.')
    if not 0 <= c_min <= 1:
        raise ValueError('c_min must be in range 0 to 1.')
    class_count = len(curvature_edges) - 1
    if curvature_classes is not None and (
        curvature_classes.shape != probabilities.shape
        or not np.issubdtype(curvature_classes.dtype, np.integer)
        or np.any((curvature_classes < -1) | (curvature_classes >= class_count))
    ):
        raise ValueError('Curvature classes must match probabilities and contain valid indices.')
    offsets = _support_offsets(
        tangent_angles, neighborhood_radius, curvature_edges, c_min
    )
    if curvature_classes is None:
        initial_support, curvature_classes = _accumulate_support(
            probabilities, offsets, class_count, None
        )
        del initial_support
    return _accumulate_support(probabilities, offsets, class_count, curvature_classes)
