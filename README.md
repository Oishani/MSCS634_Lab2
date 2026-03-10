# MSCS-634: Advanced Big Data and Data Mining
### Lab 2: Classification Using KNN and RNN Algorithms

**Name:** Oishani Ganguly
---

## Purpose

The goal of this lab is to explore and compare the performance of two distance-based classification algorithms — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — using the Wine Dataset from sklearn. Specifically, the lab covers:

- **Data Collection & Preparation:** Loading the Wine dataset, inspecting class distributions, and applying StandardScaler to normalize features before distance-based classification
- **KNN Implementation:** Training and evaluating KNN classifiers using k ∈ {1, 5, 11, 15, 21} to understand how the number of neighbors affects accuracy
- **RNN Implementation:** Training and evaluating Radius Neighbors classifiers using radius values of {350, 400, 450, 500, 550, 600} (scaled to the standardized feature space) to understand how neighborhood radius affects accuracy
- **Visualization:** Producing accuracy trend plots, confusion matrices, and per-class precision/recall charts to communicate model behavior clearly
- **Comparative Analysis:** Identifying which model and parameter setting best suits the Wine dataset and explaining the reasoning

The dataset used is the [Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) from `sklearn.datasets`, which contains 178 wine samples from three cultivars with 13 continuous chemical features each.

---

## Key Insights

### KNN Results

- **Accuracy is consistently strong across all k values**, with the best performance generally at smaller k (e.g., k=5), confirming that the Wine dataset has clean, well-separated class boundaries in the standardized feature space.
- **k=1 shows slight overfitting** — it is very sensitive to individual noisy training samples and is not the most generalizable choice despite achieving high accuracy on this small test set.
- **k=21 introduces mild underfitting** — the large neighborhood smooths over local structure, and some misclassification emerges between chemically-similar classes (class_0 and class_1).
- The confusion matrix for the best k confirms that most predictions are correct, with any errors concentrated along class boundaries where chemical profiles overlap.

### RNN Results

- **RNN accuracy rises with radius up to an optimal midpoint**, then plateaus or marginally decreases as the radius becomes large enough to include training points from competing classes.
- Because all features are standardized, the lab-specified radii (350–600) were divided by 100 to operate meaningfully in the normalized Euclidean space (3.5–6.0). This conversion is critical — without it, every training point falls within radius and all predictions collapse to a majority-class vote.
- The `outlier_label='most_frequent'` fallback was used to handle test points that may fall in low-density regions at smaller radii, preventing runtime errors while maintaining predictive coverage.
- RNN's per-class metrics are competitive with KNN's at its best radius, but the model is noticeably more sensitive to parameter choice.

### Model Comparison

- **KNN is more stable and easier to tune** on this dataset. Its accuracy varies modestly across k values, and no special handling of edge cases is required.
- **RNN is more sensitive** to the choice of radius and requires understanding of the scaled feature space to set meaningful values. On a small, uniformly-distributed dataset like Wine, this adds complexity without a performance benefit.
- **Recommendation:** KNN with k=5 is the preferred classifier for the Wine dataset — it achieves high accuracy, is straightforward to interpret, and requires no outlier-label fallback logic.

---

## Challenges and Decisions

**Feature Scaling:**
Both KNN and RNN rely entirely on Euclidean distance, making feature scaling non-negotiable. Without standardization, features with large magnitudes (e.g., `proline`, which ranges in the hundreds) would dominate the distance calculation and render other features statistically invisible. StandardScaler was fit exclusively on training data to prevent data leakage.

**Radius Conversion:**
The lab specifies radius values of 350–600. Applied directly to a standardized feature space where all values are typically in the range [-3, 3], these radii would include every training point for every prediction, reducing the model to a trivial majority-class classifier. Dividing by 100 places the radii at 3.5–6.0, which corresponds to meaningful neighborhoods in the standardized space while still allowing for variation in neighbor count.

**Outlier Label Handling:**
At smaller radii in RNN, some test points may have no training neighbors within the specified distance. `RadiusNeighborsClassifier` raises a warning and can produce errors without a fallback. Setting `outlier_label='most_frequent'` ensures graceful degradation — the most common training class is assigned rather than an exception being thrown.

**Train/Test Split Stratification:**
With only 178 samples across three classes, a simple random split risks class imbalance in the test set. `stratify=y` was used to ensure proportional class representation in both splits, making evaluation more reliable and comparable.

**Visualization Design:**
All charts were built with `matplotlib` only to ensure the notebook runs in any standard Python environment without additional dependencies.