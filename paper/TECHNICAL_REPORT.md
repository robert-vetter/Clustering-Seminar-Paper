# Automated Multi-Object Clustering Using Machine Learning

**An English technical write-up of the original German seminar paper**
*("Automatisierte Multiobjektclusterung mithilfe von Machine-Learning-Verfahren",
Seminarfacharbeit, grades 11/12, 2022–2024).*
The German original is in [`seminar-paper.pdf`](seminar-paper.pdf). This document
summarises that work faithfully in English; numbers and methods are taken from
the paper and the project notebook.

**Authors:** Hagen Jacob, Jan Edgar König, Robert Vetter
**Institution:** Spezialschulteil des Albert-Schweitzer-Gymnasiums Erfurt
**External scientific advisors:** Dr. Wolfgang Felber and Christopher Sobel,
Fraunhofer IIS (Nuremberg) — advisory role only.

---

## 1. Problem

Clustering algorithms group unlabelled data, but no single algorithm works well
for every dataset: K-Means suits compact round clusters, DBSCAN handles
density-based and non-convex shapes, OPTICS handles varying densities, and
Gaussian Mixture Models (GMM) suit overlapping, probabilistically-shaped groups.
Choosing the right one is normally a manual, expertise-dependent step.

The goal of this project was to **automate that choice**: given an arbitrary 2-D
dataset, recommend the most suitable clustering algorithm among four
complementary options — **K-Means, DBSCAN, OPTICS, GMM** — using a neural
network trained on geometric and statistical descriptions of datasets.

## 2. Background

The paper reviews supervised vs. unsupervised learning, the four clustering
algorithms above (and hierarchical clustering), and clustering-evaluation
metrics. Two metrics are used internally by the pipeline:

- **Silhouette score** — measures how well points fit their assigned cluster
  versus neighbouring clusters; used to choose between candidate shape
  descriptions.
- **Inertia** — within-cluster sum of squares (discussed as a standard metric).

## 3. Method

The system is a three-stage pipeline feeding a neural-network classifier.

### 3.1 Anomaly detection

Outliers are removed first so they do not distort the shape analysis. Two
detectors are combined:

- **DBSCAN** with `eps` chosen by a k-distance heuristic (the 93rd percentile of
  k-nearest-neighbour distances, allowing ~7% noise) and `min_samples` selected
  by silhouette score.
- **Local Outlier Factor (LOF)**, whose `n_neighbors` and `contamination` are
  grid-searched using the **macro F1-score** (appropriate for the strongly
  imbalanced outlier-vs-inlier setting).

The union of the points flagged by either method is treated as outliers.

### 3.2 Shape recognition

The geometry of the cleaned point cloud is captured by the contours that enclose
its dense regions. Two independent methods are computed and the better one is
selected by silhouette score:

- **Method 1 — density contours:** a 2-D Gaussian kernel-density estimate is
  thresholded at `factor · mean(density)` for several factors.
- **Method 2 — histogram edges:** a 2-D histogram is processed with a Sobel
  operator and a Gaussian filter, after which contours are extracted with
  `scikit-image`.

### 3.3 Feature extraction

Each dataset is reduced to a fixed **10-dimensional feature vector** describing
its shape and statistics: mean and standard deviation of contour eccentricity,
contour-area / convex-hull ratio, variance of distances to the centroid,
contour-area / points ratio, mean and standard deviation of kurtosis and of
skewness, and compactness. (Full definitions in [`../data/DATA.md`](../data/DATA.md).)

### 3.4 Classifier

A small Keras `Sequential` network maps the 10 features to one of the four
algorithms:

```
Dense(48, ReLU, L2) → Dropout(0.5) → BatchNorm
Dense(32, ReLU, L2) → Dropout(0.5) → BatchNorm
Dense(4, softmax)
```

Optimiser Adam (learning rate 1e-4), loss sparse categorical cross-entropy,
batch size 32, 100 epochs. The layer sizes were chosen by trying combinations of
hyperparameters.

## 4. Data

Training data is **synthetic**: 5,184 datasets generated with scikit-learn
(`make_blobs`, `make_circles`, `make_moons`, `make_gaussian_quantiles`), balanced
across the four classes (1,296 each), each expanded with random augmentations
(remove/add points, scale, shift, rotate). The label for each dataset is the
algorithm best suited to its generating family (blobs → K-Means, circles →
OPTICS, moons → DBSCAN, Gaussian quantiles → GMM). This labelling is a design
assumption, not the empirical result of running all four algorithms — a known
simplification. The paper notes that more generated datasets yielded higher
accuracy, which motivated the choice of 5,184.

## 5. Results

On a held-out split of the **synthetic** dataset, the network reaches
**≈99.8% accuracy** (99.81% reported in the paper; the committed notebook run
logged 99.90%). Training and validation curves are reconstructed from the real
logged run in [`../figures/07_training_curves.png`](../figures/07_training_curves.png).

**This is synthetic validation.** The companion article reports roughly **70%**
on real-world datasets, and the paper itself states the method was tested
"mainly on synthetic and only a few real datasets." The synthetic-vs-real gap is
the main caveat: the high synthetic accuracy reflects how cleanly the synthetic
families map to their labels, not production performance.

## 6. Limitations and future work

- Labels are assigned by generator family, not by empirically comparing all four
  algorithms on each dataset.
- Evaluation is dominated by synthetic data; real-world performance is markedly
  lower and only lightly tested.
- The method is restricted to **2-D** data and to **four** algorithms.

The paper suggests extending to more algorithms, applying the system to a
concrete real-world problem (e.g. collision-aware robot control), and adding a
user interface.

## 7. Acknowledgments

The work was supervised at school by Dr. Marion Moor and Johannes Süpke, and
advised externally by Dr. Wolfgang Felber and Christopher Sobel of
Fraunhofer IIS (Nuremberg), who gave regular feedback and hosted a practicum.
Fraunhofer's involvement was **scientific advice only**; the institute did not
adopt or deploy the system. The project was carried out collaboratively by the
three authors on a shared notebook, with little formal division of labour.
