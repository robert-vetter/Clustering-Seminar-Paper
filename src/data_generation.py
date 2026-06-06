"""
Synthetic dataset generation and feature table construction.

Builds the labelled training table used by the classifier. Synthetic datasets
are drawn from four scikit-learn generators, each labelled with the clustering
algorithm best suited to that structure:

    make_blobs              -> "KMeans"
    make_circles            -> "OPTICS"
    make_moons              -> "DBSCAN"
    make_gaussian_quantiles -> "GMM"

The label is therefore a *design assumption* tied to the generating
distribution, not the empirical winner of running all four algorithms. Each
dataset is additionally augmented (see ``augmentation.DataAugmentor``), the
10-dimensional feature vector is extracted for every dataset, and the resulting
table is saved both unscaled and ``RobustScaler``-scaled.

Running this regenerates ``data/data_scaled.xlsx``. It is computationally heavy
(thousands of datasets, each running the full pipeline); the committed
``data/data_scaled.xlsx`` is the artifact produced by the original run.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_gaussian_quantiles,
    make_moons,
)
from sklearn.preprocessing import RobustScaler

from .augmentation import DataAugmentor
from .feature_extraction import extract_features


def generate_datasets() -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Generate the list of ``(points, labels, best_algorithm)`` datasets."""
    warnings.filterwarnings("ignore")

    # parameters for blobs / circles
    n_samples_range = [500, 800, 1000, 1200, 1500, 1700, 2000, 2500, 4000]
    n_clusters_range = [3, 4, 5, 6]
    noise_range = [0.02, 0.04, 0.06]
    factors = [0.3, 0.5]

    # parameters for moons / gaussian quantiles
    n_samples_range_moons = [500, 800, 1000, 1200, 1500, 1700, 2000, 2500, 4000]
    n_samples_range_quantiles = [500, 800, 1000, 1200, 1500, 1700, 2000, 2500, 4000]
    noise_range_moons = [0.02, 0.04, 0.06]
    n_classes_range = [1, 2, 3]

    n_augment = 2
    n_iter_first = 2
    n_iter_second = 16

    datasets: List[Tuple[np.ndarray, np.ndarray, str]] = []

    best_algorithms_blobs = ["KMeans"] * n_iter_first
    best_algorithms_circles = ["OPTICS"] * n_iter_first
    best_algorithms_moons = ["DBSCAN"] * n_iter_second
    best_algorithms_gaussian_quantiles = ["GMM"] * n_iter_second

    # blobs + circles
    for n_samples in n_samples_range:
        for n_clusters in n_clusters_range:
            for noise in noise_range:
                for i in range(n_iter_first):
                    for j in factors:
                        random_state = np.random.randint(10000)

                        X, y = make_blobs(
                            n_samples=n_samples,
                            centers=n_clusters,
                            cluster_std=noise,
                            random_state=random_state,
                        )
                        augmentor = DataAugmentor(X)
                        for _ in range(n_augment):
                            datasets.append(
                                (augmentor.augment(), y, best_algorithms_blobs[i])
                            )
                        datasets.append((X, y, best_algorithms_blobs[i]))

                        X, y = make_circles(
                            n_samples=n_samples,
                            noise=noise,
                            factor=j,
                            random_state=random_state,
                        )
                        augmentor = DataAugmentor(X)
                        for _ in range(n_augment):
                            datasets.append(
                                (augmentor.augment(), y, best_algorithms_circles[i])
                            )
                        datasets.append((X, y, best_algorithms_circles[i]))

    # moons
    for n_samples in n_samples_range_moons:
        for noise in noise_range_moons:
            for i in range(n_iter_second):
                random_state = np.random.randint(10000)
                X, y = make_moons(
                    n_samples=n_samples, noise=noise, random_state=random_state
                )
                augmentor = DataAugmentor(X)
                for _ in range(n_augment):
                    datasets.append(
                        (augmentor.augment(), y, best_algorithms_moons[i])
                    )
                datasets.append((X, y, best_algorithms_moons[i]))

    # gaussian quantiles
    for n_samples in n_samples_range_quantiles:
        for n_classes in n_classes_range:
            for i in range(n_iter_second):
                random_state = np.random.randint(10000)
                X, y = make_gaussian_quantiles(
                    n_samples=n_samples,
                    n_features=2,
                    n_classes=n_classes,
                    random_state=random_state,
                )
                augmentor = DataAugmentor(X)
                for _ in range(n_augment):
                    datasets.append(
                        (augmentor.augment(), y, best_algorithms_gaussian_quantiles[i])
                    )
                datasets.append((X, y, best_algorithms_gaussian_quantiles[i]))

    return datasets


def build_feature_table(
    datasets: List[Tuple[np.ndarray, np.ndarray, str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features for every dataset; return (unscaled_df, scaled_df)."""
    all_unscaled_features = []
    label_list = []

    total = len(datasets)
    for counter, (x, y, best_algorithm) in enumerate(datasets, start=1):
        x = np.array(x)
        y = np.array(y)
        all_unscaled_features.append(extract_features(x, y))
        label_list.append(best_algorithm)
        print(f"Pass: {counter} / {total}")

    features_array = np.vstack(all_unscaled_features)
    df_unscaled = pd.DataFrame(
        features_array,
        columns=[f"Feature {i + 1}" for i in range(features_array.shape[1])],
    )
    df_unscaled["Best Algorithm"] = label_list

    # column-wise RobustScaler (less sensitive to outliers)
    df_scaled = df_unscaled.copy()
    scaler = RobustScaler()
    for col in df_scaled.columns:
        if col != "Best Algorithm":
            df_scaled[col] = scaler.fit_transform(
                df_scaled[col].values.reshape(-1, 1)
            ).reshape(-1)

    cols = ["Best Algorithm"] + [c for c in df_scaled if c != "Best Algorithm"]
    df_scaled = df_scaled[cols]
    return df_unscaled, df_scaled


def main(out_dir: str = "data") -> None:
    """Regenerate the dataset and write the scaled / unscaled tables."""
    datasets = generate_datasets()
    df_unscaled, df_scaled = build_feature_table(datasets)

    os.makedirs(out_dir, exist_ok=True)
    df_unscaled.to_excel(
        os.path.join(out_dir, "daten_unscaled.xlsx"), index=False, engine="openpyxl"
    )
    df_scaled.to_excel(
        os.path.join(out_dir, "data_scaled.xlsx"), index=False, engine="openpyxl"
    )
    print(f"Saved scaled/unscaled feature tables to: {out_dir}/")


if __name__ == "__main__":
    main()
