"""
Stage 3 - Feature extraction.

A cleaned point cloud is reduced to a fixed 10-dimensional feature vector that
characterises its geometry and statistics. These 10 features are the input to
the neural-network classifier.

Feature vector (in order):
    1.  mean eccentricity of the contours
    2.  std  eccentricity of the contours
    3.  mean contour-area / convex-hull-area ratio
    4.  variance of point distances to the centroid
    5.  mean contour-area / number-of-points ratio
    6.  mean kurtosis (over the two coordinates)
    7.  std  kurtosis
    8.  mean skewness
    9.  std  skewness
    10. compactness (mean squared distance to the centroid)
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import scipy.stats as stats
from shapely.geometry import Point, Polygon

from .anomaly_detection import remove_outliers
from .shape_recognition import (
    DENSITY_FACTORS,
    SIGMA_VALUES,
    find_best_factor,
    find_best_sigma,
    gaussian_density_estimation,
    select_best_contour_method,
)


def euclidean_distance(p1, p2) -> float:
    """Euclidean distance between two 2-D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def mean_std_eccentricity(contour_points: List) -> Tuple[float, float]:
    """Mean and std contour eccentricity (0 = circular, ->1 = elongated).

    Each contour with at least five points is fitted with an ellipse via
    ``cv2.fitEllipse``; eccentricity is derived from its axes.
    """
    eccentricities = []
    for contour in contour_points:
        contour_array = np.array(contour, dtype=np.float32)
        if len(contour) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(contour_array)
            if ma != 0:
                eccentricities.append(np.sqrt(1 - (MA / ma) ** 2))

    if not eccentricities:
        return 0, 0
    return np.mean(eccentricities), np.std(eccentricities)


def kurtosis_and_skewness(data: np.ndarray) -> Tuple[float, float, float, float]:
    """Mean/std kurtosis and mean/std skewness over the coordinate axes."""
    kurtosis_vals = stats.kurtosis(data)
    skewness_vals = stats.skew(data)
    return (
        np.mean(kurtosis_vals),
        np.std(kurtosis_vals),
        np.mean(skewness_vals),
        np.std(skewness_vals),
    )


def contour_to_convexhull_ratio(contour_points: List) -> float:
    """Mean ratio of contour area to its convex-hull area.

    Closer to 1 means the contour is close to its convex hull (more convex /
    circular).
    """
    ratios = []
    for contour in contour_points:
        contour_array = np.array(contour, dtype=np.float32)
        contour_area = cv2.contourArea(contour_array)
        hull = cv2.convexHull(contour_array)
        hull_area = cv2.contourArea(hull)
        ratios.append(0 if hull_area == 0 else contour_area / hull_area)
    return np.mean(ratios)


def variance_of_distances(data: np.ndarray) -> float:
    """Variance of the distances from each point to the centroid."""
    centroid = np.mean(data, axis=0)
    distances = np.sqrt(np.sum((data - centroid) ** 2, axis=1))
    return np.var(distances)


def contour_area_to_points_ratio(contours: List, data_points: np.ndarray) -> float:
    """Mean ratio of contour area to the number of points it encloses."""
    total_ratio = 0
    for contour in contours:
        contour_polygon = Polygon(contour)
        contour_area = contour_polygon.area
        points_inside = [p for p in data_points if contour_polygon.contains(Point(p))]
        points_area = len(points_inside)
        if contour_area != 0:
            total_ratio += contour_area / points_area
    return total_ratio / len(contours) if contours else 0


def centroid(points: np.ndarray) -> np.ndarray:
    """Centroid (mean) of the points."""
    return np.mean(points, axis=0)


def compactness(points: np.ndarray, center: np.ndarray) -> float:
    """Compactness: mean squared distance of the points to ``center``."""
    distances = np.linalg.norm(points - center, axis=1)
    return np.mean(distances**2)


def extract_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Run the full pipeline and return the 10-dimensional feature vector.

    Steps: remove outliers -> estimate density / contours (both methods) ->
    pick the better contour set -> compute the 10 features.
    """
    x_without_outliers, _, _, _ = remove_outliers(X, y)

    center = centroid(x_without_outliers)

    X_, Y, Z, _ = gaussian_density_estimation(x_without_outliers)

    contour_points_method1, _ = find_best_factor(
        DENSITY_FACTORS, X_, Y, Z, x_without_outliers
    )
    contour_points_method2, _ = find_best_sigma(SIGMA_VALUES, x_without_outliers)

    contour_points, _, _ = select_best_contour_method(
        contour_points_method1, contour_points_method2, x_without_outliers
    )

    mean_ecc, std_ecc = mean_std_eccentricity(contour_points)
    convex_ratio = contour_to_convexhull_ratio(contour_points)
    dist_variance = variance_of_distances(x_without_outliers)
    area_points_ratio = contour_area_to_points_ratio(contour_points, x_without_outliers)
    comp = compactness(x_without_outliers, center)
    kurt, std_kurt, skew, std_skew = kurtosis_and_skewness(X)

    return np.array(
        [
            mean_ecc,
            std_ecc,
            convex_ratio,
            dist_variance,
            area_points_ratio,
            kurt,
            std_kurt,
            skew,
            std_skew,
            comp,
        ]
    )


# Human-readable names for the 10 features, in vector order.
FEATURE_NAMES: List[str] = [
    "mean_eccentricity",
    "std_eccentricity",
    "contour_to_convexhull_ratio",
    "variance_of_distances_to_centroid",
    "contour_area_to_points_ratio",
    "mean_kurtosis",
    "std_kurtosis",
    "mean_skewness",
    "std_skewness",
    "compactness",
]
