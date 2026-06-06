"""
Stage 1 - Anomaly / outlier detection.

Outliers are removed before any shape analysis so that contour extraction and
the geometric features are not distorted by noise points. Two complementary
detectors are combined:

* **DBSCAN** with an automatically chosen ``eps`` (k-distance heuristic) and a
  ``min_samples`` value selected by silhouette score.
* **Local Outlier Factor (LOF)**, whose ``n_neighbors`` / ``contamination`` are
  grid-searched against the (synthetic) ground-truth labels using the macro
  F1-score, which is well suited to the imbalanced outlier-vs-inlier setting.

The union of the points flagged by either detector is treated as outliers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy import quantile
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def determine_epsilon(x: np.ndarray, min_samples: int) -> float:
    """Estimate DBSCAN's ``eps`` via the k-distance heuristic.

    For every point the distance to its ``min_samples``-th nearest neighbour is
    computed; ``eps`` is the 93rd percentile of those distances, i.e. at most
    ~7% of the points are allowed to fall outside a dense neighbourhood.
    """
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(x)
    distances, _ = nbrs.kneighbors(x)

    # sort and keep, per point, the distance to the farthest of its k neighbours
    distances = np.sort(distances, axis=0)
    distances = distances[:, -1]

    # allow at most ~7% noise points
    epsilon = quantile(distances, 0.93)
    return epsilon


def detect_outliers(
    x: np.ndarray, n_neighbors: int, contamination: float
) -> Tuple[int, float, np.ndarray]:
    """Combine LOF and DBSCAN to flag outliers.

    ``min_samples`` is swept over ``[2 * n_features, 3 * n_features)`` and the
    value giving the best silhouette score (ignoring DBSCAN's ``-1`` noise
    label) is kept. Returns ``(best_min_samples, best_epsilon, all_outliers)``
    where ``all_outliers`` is the union of LOF- and DBSCAN-flagged indices.
    """
    # LOF scores
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_scores = -lof_model.fit_predict(x)
    lof_thresh = np.quantile(lof_scores, 0.999)
    lof_index = np.where(lof_scores >= lof_thresh)

    min_samples_range = range(2 * x.shape[1], 3 * x.shape[1])

    best_sil_score = -1
    best_min_samples = None
    best_epsilon = None

    for min_samples in min_samples_range:
        epsilon = determine_epsilon(x, min_samples)
        model = DBSCAN(eps=epsilon, min_samples=min_samples).fit(x)
        labels = model.labels_

        # silhouette is only defined for >1 real cluster (-1 marks noise)
        if len(set(labels)) > 2:
            silhouette_avg = silhouette_score(x[labels != -1], labels[labels != -1])
            if silhouette_avg > best_sil_score:
                best_sil_score = silhouette_avg
                best_min_samples = min_samples
                best_epsilon = epsilon
        else:
            best_epsilon = epsilon
            best_min_samples = min_samples

    # final DBSCAN pass with the best parameters
    model = DBSCAN(eps=best_epsilon, min_samples=best_min_samples).fit(x)
    labels = model.labels_
    dbscan_index = np.where(labels == -1)

    # union of LOF and DBSCAN outliers
    all_outliers = np.unique(np.concatenate([dbscan_index[0], lof_index[0]]))

    return best_min_samples, best_epsilon, all_outliers


def grid_search_lof(x: np.ndarray, y: np.ndarray) -> Tuple[dict, float]:
    """Grid-search LOF's ``n_neighbors`` / ``contamination`` by macro F1.

    The macro F1-score is used because the outlier-vs-inlier classes are highly
    imbalanced, so plain accuracy would be misleading.
    """
    best_score = -1
    best_params = {"n_neighbors": None, "contamination": None}

    for n_neighbors in range(10, 25):
        for contamination in [0.01, 0.02, 0.03]:
            lof_model = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination
            )
            y_pred = lof_model.fit_predict(x)
            y_pred = [1 if v == -1 else 0 for v in y_pred]

            score = f1_score(y, y_pred, average="macro")
            if score > best_score:
                best_score = score
                best_params["n_neighbors"] = n_neighbors
                best_params["contamination"] = contamination
    return best_params, best_score


def remove_outliers(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Coordinate the full outlier-removal step.

    Returns the cleaned points and labels together with the DBSCAN ``eps`` and
    ``min_samples`` that were selected.
    """
    best_params, _ = grid_search_lof(x, y)
    best_min_samples, best_epsilon, all_outliers = detect_outliers(
        x, best_params["n_neighbors"], best_params["contamination"]
    )

    mask = np.ones(x.shape[0], dtype=bool)
    mask[all_outliers] = False
    x_without_outliers = x[mask]
    y_without_outliers = y[mask]

    return x_without_outliers, y_without_outliers, best_epsilon, best_min_samples
