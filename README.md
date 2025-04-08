# Support Vector Machine (SVM) Classification

This project focuses on implementing and optimizing **Support Vector Machines (SVM)** for classification tasks using two datasets: **Iris** (simple) and **MNIST** (complex). The goal is to explore SVM performance with linear and non-linear kernels, optimize hyperparameters, and evaluate results.

---

## Datasets

### 1. Iris Dataset
- **Source:** `sklearn.datasets.load_iris`
- **Features:** 4 numeric features (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Samples:** 150 (50 per class)
- **Purpose:** Test SVM with a linearly separable dataset.

### 2. MNIST Dataset
- **Source:** `sklearn.datasets.fetch_openml('mnist_784')`
- **Features:** 784 pixel values (28x28 grayscale images flattened)
- **Classes:** 10 (digits 0-9)
- **Samples:** 70,000 (60,000 train, 10,000 test)
- **Purpose:** Challenge SVM with a complex, non-linear dataset.

---

## Project Steps

### 1. Data Preprocessing
- **Standardization:** Features scaled using `StandardScaler` (mean=0, std=1).
- **Splitting:** 80% train, 20% test (`random_state=42`).
- **Binary Subset:** For MNIST, a binary subset (0 vs. 1) was used to test linear SVM separately.

### 2. SVM Implementation
#### Iris (Linear Kernel)
- **Settings:** `kernel='linear'`, \( C=1.0 \)
- **Approach:** Multi-class handled via One-vs-Rest (OvR) internally.

#### MNIST Binary (Linear Kernel)
- **Settings:** `kernel='linear'`, \( C=1.0 \)
- **Use Case:** Binary classification (0 vs. 1).

#### MNIST Multi-class (RBF Kernel)
- **Settings:** `kernel='rbf'`, optimized with GridSearchCV.
- **Parameters:** \( C \) in `[0.1, 1.0, 10.0]`, \( gamma \) in `['scale', 'auto', 0.1]`
- **Approach:** Multi-class via OvR.

### 3. Results
#### Iris (Linear Kernel)
- **Accuracy:** 0.9667
- **Confusion Matrix:** `[[10 0 0] [0 9 1] [0 0 10]]`
- **Analysis:** Near-perfect performance due to linear separability of the 3 classes.

#### MNIST Binary (Linear Kernel, 0 vs. 1)
- **Accuracy:** 0.9966
- **Confusion Matrix:** `[[1401 1] [1 1553]]`
- **Analysis:** Excellent results as 0 and 1 are linearly separable.

#### MNIST Multi-class (RBF Kernel, Optimized)
- **Best Parameters:** \( C=10.0 \), \( gamma='scale' \)
- **Accuracy:** 0.9721
- **Confusion Matrix:** (10x10 matrix, see code output)
- **Analysis:** Strong performance, capturing non-linear patterns better than linear models.

---

## Tools and Libraries
- **Python:** 3.x
- **Libraries:**
  - `sklearn` for SVM (`SVC`), preprocessing, and optimization
  - `numpy` for numerical operations
  - `matplotlib`, `seaborn` for visualization

## Key Code
- **Preprocessing:** `StandardScaler`, `train_test_split`
- **Model:** `SVC` with `kernel='linear'` and `kernel='rbf'`
- **Optimization:** `GridSearchCV` (3-fold CV, `n_jobs=-1`)
- **Evaluation:** `accuracy_score`, `confusion_matrix`, `classification_report`

## Results Summary
| Dataset       | Kernel    | Accuracy | Best Parameters          | Notes                          |
|---------------|-----------|----------|--------------------------|--------------------------------|
| Iris          | Linear    | 0.9667   | \( C=1.0 \)              | Near-perfect, linearly separable |
| MNIST (0 vs 1)| Linear    | 0.9966   | \( C=1.0 \)              | Excellent for binary task      |
| MNIST (0-9)   | RBF       | 0.9721   | \( C=10.0, gamma='scale' \) | Captures non-linear patterns  |

## Key Insights
- **Linear SVM:** Highly effective for simple, separable datasets (Iris, MNIST 0 vs. 1).
- **RBF SVM:** Superior for complex, non-linear data (MNIST 0-9), achieving 97% accuracy.
- **Multi-class:** SVM handles multi-class problems natively via OvR, no manual binary conversion needed.

## Next Steps
1. Test additional \( C \) and \( gamma \) values for RBF kernel.
2. Analyze misclassified MNIST digits to understand errors.
3. Experiment with other kernels (e.g., polynomial) for comparison.
