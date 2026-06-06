"""
Stage 2 - Shape / contour recognition.

The overall geometry of a (cleaned) point cloud is captured by extracting the
contours that enclose its dense regions. Two independent methods are computed
and the better one is chosen by silhouette score:

* **Method 1 - density contours:** a 2-D Gaussian kernel-density estimate is
  thresholded at ``factor * mean(density)`` for several factors; the factor with
  the best silhouette score wins.
* **Method 2 - histogram edges:** a 2-D histogram is run through a Sobel
  operator and a Gaussian filter, then ``skimage`` extracts contours; the
  Gaussian ``sigma`` with the best silhouette score wins.

A point is assigned to the contour (polygon) that contains it; points in no
polygon are treated as noise when scoring.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon
from skimage import measure
from sklearn.metrics import silhouette_score

# Defaults swept by the search routines (kept identical to the original notebook).
DENSITY_FACTORS: List[float] = [0.7, 1.0, 1.5, 3.0]
SIGMA_VALUES: List[float] = [0.7, 1, 1.5, 3]
PADDING_PERCENTAGE: float = 0.2


# --------------------------------------------------------------------------- #
# Shared scoring
# --------------------------------------------------------------------------- #
def contour_silhouette_score(
    contour_points: List, data_points: np.ndarray
) -> Tuple[float, list]:
    """Label each point by the contour that contains it and score the result.

    Contours are simplified for speed. Each data point gets the index of the
    first polygon that contains it, or ``-1`` if it is outside every polygon.
    Returns ``(silhouette_score, labels)`` and ``-1`` if fewer than two labels.
    """
    simplified_polygons = [
        Polygon(contour).simplify(0.1, preserve_topology=False)
        for contour in contour_points
    ]
    cluster_labels = []
    for point in data_points:
        for i, polygon in enumerate(simplified_polygons):
            if polygon.contains(Point(point)):
                cluster_labels.append(i)
                break
        else:
            cluster_labels.append(-1)

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2:
        return -1, cluster_labels

    return silhouette_score(data_points, cluster_labels), cluster_labels


# --------------------------------------------------------------------------- #
# Method 1 - Gaussian density estimation
# --------------------------------------------------------------------------- #
def gaussian_density_estimation(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a 2-D density grid over the data via a Gaussian KDE."""
    density = gaussian_kde(data.T)
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.reshape(density(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
    Z_smooth = gaussian_filter(Z, sigma=1)
    return X, Y, Z, Z_smooth


def get_contours_with_factor(
    factor: float, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> List:
    """Extract density contours at the level ``factor * mean(Z)``."""
    mean_z = np.mean(Z.ravel())

    # if the threshold exceeds every density value there are no contours
    if np.all(factor * mean_z > Z):
        return []

    fig, ax = plt.subplots()
    contours = ax.contour(X, Y, Z, [factor * mean_z])
    plt.close(fig)

    if not contours.get_paths():
        return []

    contour_paths = contours.get_paths()
    if not contour_paths:
        return []

    all_contour_points = []
    for path in contour_paths:
        vertices = path.vertices
        all_contour_points.append(vertices.tolist())
    return all_contour_points


def find_best_factor(
    factors: List[float],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    data_points: np.ndarray,
) -> Tuple[Optional[List], Optional[float]]:
    """Pick the density factor whose contours maximise the silhouette score."""
    best_factor = None
    best_silhouette_score = -1
    best_contour_points = None

    for factor in factors:
        contour_points = get_contours_with_factor(factor, X, Y, Z)
        if not contour_points:
            continue

        silhouette, _ = contour_silhouette_score(contour_points, data_points)
        if silhouette >= best_silhouette_score:
            best_silhouette_score = silhouette
            best_factor = factor
            best_contour_points = contour_points

    return best_contour_points, best_factor


# --------------------------------------------------------------------------- #
# Method 2 - histogram + Sobel + Gaussian filter
# --------------------------------------------------------------------------- #
def create_histogram(
    data: np.ndarray, padding_percentage: float, bins: List[int] = [64, 64]
) -> Tuple[np.ndarray, float, float, float, float]:
    """Build a padded 2-D histogram of the points."""
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * padding_percentage
    x_max += x_range * padding_percentage
    y_min -= y_range * padding_percentage
    y_max += y_range * padding_percentage

    hist, _, _ = np.histogram2d(
        data[:, 0], data[:, 1], bins=bins, range=[[x_min, x_max], [y_min, y_max]]
    )
    return hist, x_min, x_max, y_min, y_max


def process_histogram(hist: np.ndarray, sigma: float) -> Tuple[np.ndarray, list]:
    """Apply a Sobel operator and Gaussian filter, then extract contours."""
    dx = ndimage.sobel(hist, 0)
    dy = ndimage.sobel(hist, 1)
    mag = np.hypot(dx, dy)

    smoothed_mag = gaussian_filter(mag, sigma=sigma)
    binary_image = smoothed_mag > 1
    contours = measure.find_contours(binary_image, 0.8)
    return smoothed_mag, contours


def scale_contours(
    contours: list,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bins: List[int] = [64, 64],
) -> List:
    """Map histogram-space contours back to data coordinates."""
    scaled_contours = []
    for contour in contours:
        scaled_contour = np.empty_like(contour)
        scaled_contour[:, 0] = x_min + contour[:, 0] * (x_max - x_min) / bins[0]
        scaled_contour[:, 1] = y_min + contour[:, 1] * (y_max - y_min) / bins[1]
        scaled_contours.append(scaled_contour)

    return [[[p[0], p[1]] for p in contour] for contour in scaled_contours]


def histogram_contours(data_points: np.ndarray, sigma: float) -> List:
    """Convenience wrapper: histogram -> edges -> scaled contours."""
    hist, x_min, x_max, y_min, y_max = create_histogram(data_points, PADDING_PERCENTAGE)
    _, contours = process_histogram(hist, sigma)
    return scale_contours(contours, x_min, x_max, y_min, y_max)


def find_best_sigma(
    sigma_values: List[float], data_points: np.ndarray
) -> Tuple[List, Optional[float]]:
    """Pick the Gaussian ``sigma`` whose contours maximise the silhouette score."""
    best_sigma = None
    best_silhouette_score = -1
    best_contour_points = None

    for sigma in sigma_values:
        contour_points_method2 = histogram_contours(data_points, sigma)
        score, _ = contour_silhouette_score(contour_points_method2, data_points)
        if score > best_silhouette_score:
            best_silhouette_score = score
            best_sigma = sigma
            best_contour_points = contour_points_method2

    # fall back to a default sigma if nothing scored
    if best_sigma is None:
        best_contour_points = histogram_contours(data_points, 1.5)

    # drop contours that contain no points
    valid_contours = []
    for contour in best_contour_points:
        polygon = Polygon(contour)
        if any(polygon.contains(Point(p[0], p[1])) for p in data_points):
            valid_contours.append(contour)

    return valid_contours, best_sigma


# --------------------------------------------------------------------------- #
# Method selection
# --------------------------------------------------------------------------- #
def select_best_contour_method(
    contour_points_method1: List,
    contour_points_method2: List,
    data_points: np.ndarray,
) -> Tuple[List, str, list]:
    """Return the contours of whichever method has the higher silhouette score."""
    silhouette_score1, labels1 = contour_silhouette_score(
        contour_points_method1, data_points
    )
    silhouette_score2, labels2 = contour_silhouette_score(
        contour_points_method2, data_points
    )

    if silhouette_score1 > silhouette_score2:
        return contour_points_method1, "method1", labels1
    return contour_points_method2, "method2", labels2


# --------------------------------------------------------------------------- #
# Plotting helpers (unchanged from the notebook)
# --------------------------------------------------------------------------- #
def plot_contour(best_contour_points: List, X, Y, Z) -> None:
    """Plot the contour lines of method 1."""
    plt.figure()
    for contour in best_contour_points:
        x, y = zip(*contour)
        plt.plot(x, y, color="red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Contour Plot Method 1")
    plt.show()


def plot_contours_over_data(contour_points: List, original_data, title: str) -> None:
    """Scatter the data and overlay contour lines."""
    plt.scatter(
        [p[0] for p in original_data], [p[1] for p in original_data], color="blue"
    )
    for contour in contour_points:
        plt.plot(
            [p[0] for p in contour], [p[1] for p in contour], color="red"
        )
    plt.title(title)
    plt.show()


def plot_filled_contours(contours: List, original_data, title: str) -> None:
    """Scatter the data and overlay filled contours."""
    plt.scatter(
        [p[0] for p in original_data], [p[1] for p in original_data], color="blue"
    )
    for contour in contours:
        plt.fill(
            [p[0] for p in contour], [p[1] for p in contour], color="red", alpha=0.3
        )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
