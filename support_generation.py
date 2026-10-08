from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from convolution import ORIENTATION_COUNT

type Position = Sequence[float] | NDArray[np.number]


def gamma(beta: float, gamma: float) -> float:
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
    return c_min


def default_curvature_edges(
    neighborhood_radius: int,
    max_curvature: float = 0.2,
    class_count: int = 7,
) -> NDArray[np.float64]:
    if class_count == 1:
        return np.array([-max_curvature, max_curvature])
    straight_limit = min(2 / neighborhood_radius ** 2, max_curvature / 2)
    positive_edges = np.linspace(straight_limit, max_curvature, class_count // 2 + 1)
    return np.concatenate((-positive_edges[::-1], positive_edges))


@dataclass(frozen=True)
class SupportOffset:
    dy: int
    dx: int
    labels: NDArray[np.int64]
    target_classes: NDArray[np.int64]
    neighbor_classes: NDArray[np.int64]
    coefficients: NDArray[np.float64]


def classify_curvatures(
    curvatures: NDArray[np.float64], edges: NDArray[np.float64]
) -> NDArray[np.int64]:
    indices = np.searchsorted(edges, curvatures, side="right") - 1
    indices[curvatures == edges[-1]] = len(edges) - 2
    indices[(curvatures < edges[0]) | (curvatures > edges[-1])] = -1
    return indices.astype(np.int64)


def support_offsets(
    angles: NDArray[np.float64],
    radius: int,
    edges: NDArray[np.float64],
    c_min: float,
) -> list[SupportOffset]:
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance_squared = dx * dx + dy * dy
            if distance_squared == 0 or distance_squared > radius * radius:
                continue
            chord_angle = np.arctan2(dy, dx) % np.pi
            interior_angles = chord_angle - angles
            interior_angles[interior_angles > np.pi / 2] -= np.pi
            interior_angles[interior_angles < -np.pi / 2] += np.pi
            tolerance = np.pi / ORIENTATION_COUNT + 2 * np.arcsin(
                1 / np.sqrt(distance_squared)
            )
            coefficients = np.where(
                np.abs(interior_angles[:, None] + interior_angles[None, :]) < tolerance,
                1.0,
                c_min,
            )
            curvatures = 2 * (np.cos(angles) * dy - np.sin(angles) * dx) / distance_squared
            target_classes = classify_curvatures(curvatures, edges)
            neighbor_classes = classify_curvatures(-curvatures, edges)
            labels = np.flatnonzero(target_classes >= 0)
            if labels.size:
                offsets.append(SupportOffset(
                    dy, dx, labels, target_classes[labels], neighbor_classes,
                    coefficients[labels],
                ))
    return offsets


def accumulate_support(
    probabilities: NDArray[np.float64],
    offsets: list[SupportOffset],
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
    class_count = len(curvature_edges) - 1
    offsets = support_offsets(
        tangent_angles, neighborhood_radius, curvature_edges, c_min
    )
    if curvature_classes is None:
        initial_support, curvature_classes = accumulate_support(
            probabilities, offsets, class_count, None
        )
        del initial_support
    return accumulate_support(probabilities, offsets, class_count, curvature_classes)
