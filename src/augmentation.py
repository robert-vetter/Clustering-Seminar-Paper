"""
Data augmentation for synthetic datasets.

``DataAugmentor`` applies random, label-preserving transformations to a 2-D
point cloud (remove/add points, scale, shift, rotate about the centroid). It is
used during dataset generation to multiply the number of training examples and
make the extracted features more robust to such variations.
"""

from __future__ import annotations

import numpy as np


class DataAugmentor:
    """Apply random label-preserving transformations to a 2-D point cloud."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.n_samples, self.n_features = data.shape

    def remove_points(self, n: int) -> np.ndarray:
        """Remove ``n`` random points."""
        indices = np.random.choice(self.n_samples, size=n, replace=False)
        return np.delete(self.data, indices, axis=0)

    def add_points(self, n: int) -> np.ndarray:
        """Add ``n`` jittered copies of random existing points."""
        indices = np.random.choice(self.n_samples, size=n)
        additional_points = self.data[indices] + 0.05 * np.random.randn(
            n, self.n_features
        )
        return np.vstack([self.data, additional_points])

    def scale_data(self, scale_factor: float) -> np.ndarray:
        """Scale all coordinates by ``scale_factor``."""
        return self.data * scale_factor

    def shift_points(self, shift_factor: float) -> np.ndarray:
        """Shift each point by a small random offset bounded by ``shift_factor``."""
        shifts = (
            (np.random.rand(self.n_samples, self.n_features) - 0.5) * 2 * shift_factor
        )
        return self.data + shifts

    def rotate_around_centroid(self, theta: float) -> np.ndarray:
        """Rotate the cloud by ``theta`` radians about its centroid (2-D only)."""
        centroid = np.mean(self.data, axis=0)
        shifted_data = self.data - centroid
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated_data = np.dot(shifted_data, R.T)
        return rotated_data + centroid

    def augment(self) -> np.ndarray:
        """Apply a random sequence of the above transformations.

        Note: each step rebinds ``result_data`` but operates on ``self.data``
        (preserved from the original notebook for reproducibility).
        """
        result_data = self.data.copy()

        remove_factor = np.random.randint(1, self.n_samples)
        result_data = self.remove_points(remove_factor)

        add_factor = np.random.randint(1, self.n_samples)
        result_data = self.add_points(add_factor)

        scale_factor = 0.5 + np.random.rand()
        result_data = self.scale_data(scale_factor)

        shift_factor = 0.05
        result_data = self.shift_points(shift_factor)

        theta = np.random.rand() * 2 * np.pi
        result_data = self.rotate_around_centroid(theta)

        return result_data
