"""
Automated clustering-algorithm selection.

A modular extraction of the original seminar-project notebook
(``notebooks/clustering-algorithm-selection.ipynb``). The package mirrors the
notebook's logic; the notebook remains the canonical, runnable reference.

Pipeline:
    1. anomaly_detection   - remove outliers (DBSCAN + Local Outlier Factor)
    2. shape_recognition   - extract dataset contours (KDE and histogram methods)
    3. feature_extraction  - compute 10 geometric/statistical features
    4. model / train       - neural network that selects the clustering algorithm
"""

__all__ = [
    "anomaly_detection",
    "shape_recognition",
    "feature_extraction",
    "augmentation",
    "data_generation",
    "model",
]
