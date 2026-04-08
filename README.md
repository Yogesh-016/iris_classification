# 🌸 Iris Flower Species Classification — Machine Learning Pipeline


A complete end-to-end machine learning pipeline to classify Iris flower species (**Setosa**, **Versicolor**, **Virginica**) using petal and sepal measurements. Six classification models are trained, compared, and evaluated.

---

## 📁 Project Structure

```
iris-classification/
│
├── Iris.csv                      # Dataset (150 samples, 6 columns)
├── iris_ml_classification.py     # Main ML pipeline script
├── iris_pairplot.png             # EDA — Pairplot by species
├── iris_correlation.png          # EDA — Feature correlation heatmap
├── iris_boxplots.png             # EDA — Boxplots per feature
├── iris_model_comparison.png     # CV accuracy comparison chart
├── iris_confusion_matrix.png     # Random Forest confusion matrix
├── iris_feature_importance.png   # Feature importance chart
└── README.md
```

---

## 📊 Dataset

| Column         | Description                  |
|----------------|------------------------------|
| `Id`           | Row identifier (dropped)     |
| `SepalLengthCm`| Sepal length in centimetres  |
| `SepalWidthCm` | Sepal width in centimetres   |
| `PetalLengthCm`| Petal length in centimetres  |
| `PetalWidthCm` | Petal width in centimetres   |
| `Species`      | Target label (3 classes)     |

- **Total samples:** 150 (50 per class)
- **Classes:** `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`
- **Missing values:** None

---

## 🔧 Tech Stack

- **Python 3.8+**
- **pandas** — data loading & manipulation
- **NumPy** — numerical operations
- **Matplotlib & Seaborn** — data visualisation
- **scikit-learn** — ML models, preprocessing, evaluation

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/iris-classification.git
cd iris-classification
```

### 2. Install dependencies
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

### 3. Run the pipeline
```bash
python iris_ml_classification.py
```

---

## 🤖 Models Trained & Compared

| Model                | Notes                              |
|----------------------|------------------------------------|
| Logistic Regression  | Linear baseline                    |
| Decision Tree        | Interpretable rule-based model     |
| **Random Forest** ✅ | Best performer — selected as final |
| Gradient Boosting    | Powerful ensemble method           |
| SVM (RBF kernel)     | Excellent for small datasets       |
| K-Nearest Neighbours | Distance-based, k=5                |

All models are evaluated using **5-fold cross-validation**.

---

## 📈 Pipeline Steps

1. **Load Data** — reads `Iris.csv`, drops `Id` column
2. **EDA** — pairplot, heatmap, and boxplots per species
3. **Preprocessing** — label encoding, 80/20 stratified split, StandardScaler
4. **Model Training** — trains 6 classifiers with cross-validation
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix
6. **Feature Importance** — identifies most predictive features
7. **Prediction** — classifies new unseen flower measurements

---

## 🏆 Results (Random Forest — Test Set)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~96–100% |
| Precision | High across all 3 classes |
| Recall    | High across all 3 classes |

> `PetalLengthCm` and `PetalWidthCm` are the most important features.

---

## 🔮 Predict New Samples

```python
new_samples = pd.DataFrame({
    'SepalLengthCm': [5.1, 6.5, 7.2],
    'SepalWidthCm' : [3.5, 2.8, 3.0],
    'PetalLengthCm': [1.4, 4.6, 5.8],
    'PetalWidthCm' : [0.2, 1.5, 2.2],
})
# Output:
# Sample 1: Iris-setosa      (confidence ~100%)
# Sample 2: Iris-versicolor  (confidence ~90%)
# Sample 3: Iris-virginica   (confidence ~95%)
```

## 🙋‍♂️ Author

Yogesh S

---

