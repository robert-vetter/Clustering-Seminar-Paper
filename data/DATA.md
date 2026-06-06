# Dataset

`data_scaled.xlsx` is the training table for the classifier. It is **synthetic**
— there is no real-world or personal data in it.

## Shape

- **5,184 rows** (one row per generated dataset)
- **11 columns**: `Best Algorithm` (label) + `Feature 1` … `Feature 10`
- **Perfectly balanced** across the four classes: 1,296 examples each of
  `KMeans`, `DBSCAN`, `OPTICS`, `GMM`

## How it was generated

Each row corresponds to one synthetic 2-D dataset drawn from a scikit-learn
generator, optionally transformed by a random augmentation
(`src/augmentation.py`: remove/add points, scale, shift, rotate). The full
pipeline (`src/data_generation.py`) then runs anomaly detection, shape
recognition, and feature extraction on each dataset to produce its 10-feature
row. Feature columns are scaled column-wise with scikit-learn's `RobustScaler`.

### Labels are assigned by generator family

The `Best Algorithm` label is **a design assumption tied to the generating
distribution**, not the empirical winner of running all four algorithms:

| Generator                 | Structure              | Label    |
|---------------------------|------------------------|----------|
| `make_blobs`              | compact round clusters | `KMeans` |
| `make_circles`            | concentric rings       | `OPTICS` |
| `make_moons`              | interleaving crescents | `DBSCAN` |
| `make_gaussian_quantiles` | nested Gaussian shells | `GMM`    |

This mapping reflects which algorithm is best suited to each structure by
construction. It is a simplification and a known limitation (see the technical
report).

## The 10 features

| # | Feature | Meaning |
|---|---------|---------|
| 1 | `mean_eccentricity` | mean eccentricity of the dataset's contours (0 = circular, →1 = elongated) |
| 2 | `std_eccentricity` | standard deviation of contour eccentricity |
| 3 | `contour_to_convexhull_ratio` | mean ratio of contour area to its convex-hull area |
| 4 | `variance_of_distances_to_centroid` | variance of point distances to the centroid |
| 5 | `contour_area_to_points_ratio` | mean ratio of contour area to number of enclosed points |
| 6 | `mean_kurtosis` | mean kurtosis over the two coordinate axes |
| 7 | `std_kurtosis` | standard deviation of kurtosis |
| 8 | `mean_skewness` | mean skewness over the two coordinate axes |
| 9 | `std_skewness` | standard deviation of skewness |
| 10 | `compactness` | mean squared distance of points to the centroid |

## Regenerating

```bash
python -m src.data_generation   # writes data/data_scaled.xlsx (+ unscaled)
```

Regeneration is computationally heavy (thousands of datasets, each run through
the full pipeline) and uses random states, so an exact byte-for-byte match of
the committed file is not expected. The committed `data_scaled.xlsx` is the
artifact from the original project run.
