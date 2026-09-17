# Python Libraries for Machine Learning

A collection of Python scripts/notebooks covering the core libraries used in data science and machine learning — from data handling and visualization to building and evaluating ML models.

## 📁 Repository Structure

```
.
├── numpy/            # Arrays, vectorized ops, linear algebra
├── pandas/           # DataFrames, cleaning, EDA
├── matplotlib/       # Static plotting basics
├── seaborn/          # Statistical visualization
├── scikit-learn/     # ML algorithms & model pipelines
├── datasets/         # Sample/toy datasets used across notebooks
└── requirements.txt
```

## 🛠️ Libraries Covered

| Library | Purpose |
|---|---|
| **NumPy** | Array creation, indexing, broadcasting, linear algebra, random sampling |
| **Pandas** | Series/DataFrame ops, merging, groupby, missing data, feature engineering |
| **Matplotlib** | Line/bar/scatter/hist plots, subplots, styling |
| **Seaborn** | Heatmaps, pairplots, distplots, categorical plots |
| **Scikit-learn** | Preprocessing, train/test split, model training, evaluation, pipelines |

## 🤖 ML Algorithms Implemented

**Supervised**
- Linear Regression, Logistic Regression
- Decision Tree, Random Forest
- KNN, SVM
- Naive Bayes
- Gradient Boosting (XGBoost/LightGBM if included)

**Unsupervised**
- K-Means Clustering
- PCA (Dimensionality Reduction)
- Hierarchical Clustering

**Model Evaluation**
- Confusion Matrix, Accuracy/Precision/Recall/F1
- Cross-validation, GridSearchCV
- ROC-AUC curves

## ⚙️ Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## 🚀 Usage

Each folder contains standalone scripts/notebooks — open and run individually:

```bash
jupyter notebook
```

or run a script directly:

```bash
python numpy/basics.py
```

## 📌 Notes

- Beginner → intermediate level, meant for learning and quick reference.
- Each script includes inline comments explaining the concept.
- Contributions/PRs for new algorithms or cleaner examples are welcome.

## 📄 License

MIT License — free to use and modify.
