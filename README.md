# Automated Multi-Object Clustering Using Machine Learning

A system that **automatically selects the most suitable clustering algorithm**
— K-Means, DBSCAN, OPTICS, or GMM — for a given 2-D dataset. A three-stage
pipeline reduces any dataset to 10 geometric/statistical features, and a small
neural network maps those features to the recommended algorithm.

Developed as a two-year school research project (*Seminarfacharbeit*, grades
11/12, 2022–2024) at the Albert-Schweitzer-Gymnasium Erfurt, with **Fraunhofer
IIS (Nuremberg) as external scientific advisors**.

![Pipeline](figures/00_pipeline.png)

> **A note on scope.** Results below are on **synthetic** data. The companion
> article reports ~70% on real-world datasets; this was a student research
> project, not a production system. See [Results](#results).

---

## Problem

No single clustering algorithm works well for every dataset — the right choice
depends on the data's shape and density, and is usually made manually by an
analyst. This project automates that decision for four complementary algorithms.

## Method

The system works in stages (`src/`):

1. **Anomaly detection** — remove outliers using **DBSCAN** (auto-tuned `eps`)
   combined with the **Local Outlier Factor** (grid-searched by macro F1).
2. **Shape recognition** — describe the dataset's geometry via its contours,
   using two independent methods and keeping the better one (by silhouette
   score): a **Gaussian kernel-density** estimate, and a **histogram + Sobel +
   Gaussian-filter** edge method.
3. **Feature extraction** — reduce the dataset to **10 features** (eccentricity,
   convex-hull ratio, kurtosis, skewness, compactness, …; see
   [`data/DATA.md`](data/DATA.md)).
4. **Classification** — a Keras neural network
   (`Dense 48 → 32 → 4 softmax`, Adam, 100 epochs) predicts the best algorithm.

| Shape recognition — method 1 (density) | Shape recognition — method 2 (histogram) |
|---|---|
| ![m1](figures/05_shape_recognition_method1.png) | ![m2](figures/06_shape_recognition_method2.png) |

*Contours are extracted around the dense regions of a (cleaned) point cloud and
used to characterise its shape.*

## Dataset

Training data is **synthetic**: **5,184** datasets generated with scikit-learn
(`make_blobs`, `make_circles`, `make_moons`, `make_gaussian_quantiles`),
**balanced** across the four classes (1,296 each), each expanded with random
augmentations. Each dataset is labelled with the algorithm best suited to its
generating structure (a design assumption — details and caveats in
[`data/DATA.md`](data/DATA.md)).

## Results

On a held-out split of the **synthetic** dataset, the classifier reaches
**99.81% accuracy**, with near-uniform per-class F1 ≈ 0.998 (two
misclassifications out of 1,037). Results are from a reproducible run (seed 42)
on the committed data; the original notebook independently logged ≈99.90%.

| Training curves | Confusion matrix |
|---|---|
| ![Training curves](figures/07_training_curves.png) | ![Confusion matrix](figures/08_confusion_matrix.png) |

*Left: training/validation accuracy and loss over 100 epochs. Right: confusion
matrix on the synthetic test split. Both from the seed-42 reproduction run.*

**This is synthetic validation, not real-world performance.** The companion
article reports roughly **70%** on real-world datasets, and the paper notes the
method was tested mainly on synthetic and only a few real datasets. The gap
between synthetic and real performance is the main limitation.

📄 **Full details:** see the [**Technical Report**](paper/TECHNICAL_REPORT.md)
(methodology, equations, experiments, per-class metrics, references).

## How to run

```bash
pip install -r requirements.txt

# 1) (optional) regenerate the synthetic dataset — heavy
python -m src.data_generation

# 2) train the classifier on data/data_scaled.xlsx
python -m src.train

# 3) evaluate on the held-out synthetic test split (+ confusion matrix)
python -m src.evaluate
```

The original, runnable notebook is preserved in
[`notebooks/clustering-algorithm-selection.ipynb`](notebooks/clustering-algorithm-selection.ipynb).
The feature pipeline (steps 1–3 of the method) runs without TensorFlow; only the
network needs it.

## Repository structure

```
.
├── src/         # modular pipeline: anomaly_detection, shape_recognition,
│                #   feature_extraction, augmentation, data_generation,
│                #   model, train, evaluate
├── notebooks/   # original project notebook (preserved)
├── data/        # data_scaled.xlsx + DATA.md (synthetic data documentation)
├── figures/     # pipeline diagram, shape-recognition plots, training curves
├── paper/       # German seminar paper (PDF) + English TECHNICAL_REPORT.md
├── requirements.txt
├── CITATION.cff
└── LICENSE      # MIT (covers the source code)
```

## Documentation

- **English technical report:** [`paper/TECHNICAL_REPORT.md`](paper/TECHNICAL_REPORT.md)
- **German seminar paper (original):** [`paper/seminar-paper.pdf`](paper/seminar-paper.pdf)
- **Dataset details:** [`data/DATA.md`](data/DATA.md)
- **Article:** [Automating clustering-algorithm selection with neural networks](https://medium.com/@robertvetter793/revolutionizing-data-science-how-neural-networks-are-automating-clustering-algorithm-selection-477c05cf9888) (self-published)

## Recognition

This project received:

- **1st Prize, Jugend forscht** (regional)
- **Fraunhofer IDMT Special Prize**
- **Albert Schweitzer School Prize** (Mathematics & Computer Science)
- **1st Prize, Inverso Software Challenge** (€1,000+ in prize money)

## Authors and acknowledgments

A team project by **Hagen Jacob, Jan Edgar König, and Robert Vetter**, carried
out collaboratively.

Supervised at school by **Dr. Marion Moor** and **Johannes Süpke**, and advised
externally by **Dr. Wolfgang Felber** and **Christopher Sobel** of **Fraunhofer
IIS (Nuremberg)**, who provided regular scientific feedback and hosted a
practicum. Fraunhofer's role was advisory only.

## License

Source code: [MIT](LICENSE). The seminar paper PDF is the authors' academic work,
included for reference.
