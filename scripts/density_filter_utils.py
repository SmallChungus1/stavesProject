from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DensityFilterResult:
    keep_indices: list[int]
    labels: np.ndarray
    eps: float
    min_samples: int
    primary_label: int | None
    cluster_sizes: dict[int, int]


def _pairwise_squared_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sum(diff * diff, axis=2)


def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Minimal DBSCAN implementation tailored for small detection sets.
    """
    point_count = len(points)
    if point_count == 0:
        return np.empty(0, dtype=int)

    eps_sq = float(eps) * float(eps)
    dist_sq = _pairwise_squared_distances(points)
    neighbors = [np.flatnonzero(dist_sq[idx] <= eps_sq).tolist() for idx in range(point_count)]

    labels = np.full(point_count, -99, dtype=int)
    visited = np.zeros(point_count, dtype=bool)
    cluster_id = 0

    for point_idx in range(point_count):
        if visited[point_idx]:
            continue

        visited[point_idx] = True
        point_neighbors = neighbors[point_idx]

        if len(point_neighbors) < min_samples:
            labels[point_idx] = -1
            continue

        labels[point_idx] = cluster_id
        seeds = set(point_neighbors)
        seeds.discard(point_idx)

        while seeds:
            current_idx = seeds.pop()

            if not visited[current_idx]:
                visited[current_idx] = True
                current_neighbors = neighbors[current_idx]
                if len(current_neighbors) >= min_samples:
                    seeds.update(current_neighbors)

            if labels[current_idx] in (-99, -1):
                labels[current_idx] = cluster_id

        cluster_id += 1

    return labels


def estimate_density_params(
    boxes: Iterable[Iterable[float]],
    image_width: int,
    image_height: int,
) -> tuple[float, int]:
    boxes_array = np.asarray(list(boxes), dtype=float)
    if boxes_array.size == 0:
        base_eps = min(image_width, image_height) * 0.035
        return float(base_eps), 3

    widths = np.clip(boxes_array[:, 2], 1.0, None)
    heights = np.clip(boxes_array[:, 3], 1.0, None)
    avg_diag = float(np.mean(np.hypot(widths, heights)))
    avg_span = float(np.mean(np.maximum(widths, heights)))

    base_eps = min(image_width, image_height) * 0.035
    adaptive_eps = max(base_eps, avg_diag * 1.25, avg_span * 1.35)
    eps_cap = min(image_width, image_height) * 0.12
    eps = float(min(adaptive_eps, eps_cap))

    point_count = len(boxes_array)
    if point_count < 4:
        min_samples = 3
    else:
        min_samples = int(max(3, min(8, ceil(point_count * 0.06))))

    return eps, min_samples


def select_primary_density_cluster(
    boxes: list[list[float]],
    scores: list[float],
    image_width: int,
    image_height: int,
) -> DensityFilterResult:
    """
    Return the indices that belong to the densest stave cluster.

    If the detections are too sparse to form a meaningful cluster, the
    original set is returned unchanged.
    """
    if len(boxes) < 3:
        return DensityFilterResult(
            keep_indices=list(range(len(boxes))),
            labels=np.full(len(boxes), -1, dtype=int),
            eps=0.0,
            min_samples=0,
            primary_label=None,
            cluster_sizes={},
        )

    boxes_array = np.asarray(boxes, dtype=float)
    centers = np.column_stack(
        (boxes_array[:, 0] + boxes_array[:, 2] / 2.0, boxes_array[:, 1] + boxes_array[:, 3] / 2.0)
    )

    eps, min_samples = estimate_density_params(boxes_array, image_width, image_height)
    labels = dbscan(centers, eps=eps, min_samples=min_samples)

    cluster_sizes: dict[int, int] = {}
    cluster_score_sums: dict[int, float] = {}
    for idx, label in enumerate(labels):
        if label < 0:
            continue
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        cluster_score_sums[label] = cluster_score_sums.get(label, 0.0) + float(scores[idx])

    if not cluster_sizes:
        return DensityFilterResult(
            keep_indices=[],
            labels=labels,
            eps=eps,
            min_samples=min_samples,
            primary_label=None,
            cluster_sizes=cluster_sizes,
        )

    primary_label = max(
        cluster_sizes,
        key=lambda label: (cluster_sizes[label], cluster_score_sums.get(label, 0.0), -label),
    )
    keep_indices = [idx for idx, label in enumerate(labels) if label == primary_label]

    return DensityFilterResult(
        keep_indices=keep_indices,
        labels=labels,
        eps=eps,
        min_samples=min_samples,
        primary_label=primary_label,
        cluster_sizes=cluster_sizes,
    )
