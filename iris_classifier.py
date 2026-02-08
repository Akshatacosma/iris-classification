import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

# Load data
iris = datasets.load_iris()
X = iris.data[:, :2]   # only 2 features for plotting
y = iris.target

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Models to compare
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Gaussian Process": GaussianProcessClassifier(kernel=RBF()),
    "Gradient Boosting": HistGradientBoostingClassifier(),
}

# Plot setup
fig, axes = plt.subplots(len(classifiers), 4, figsize=(12, 8))

# Loop over models
for row, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    for cls in range(3):
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X_train,
            response_method="predict_proba",
            class_of_interest=cls,
            ax=axes[row, cls],
            cmap="Blues",
        )
        axes[row, cls].set_title(f"{name} - Class {cls}")
        axes[row, cls].set_xticks([])
        axes[row, cls].set_yticks([])

    # Max class plot
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        response_method="predict",
        ax=axes[row, 3],
    )
    axes[row, 3].set_title(f"Max Class (Acc={acc:.2f})")
    axes[row, 3].set_xticks([])
    axes[row, 3].set_yticks([])

plt.tight_layout()
plt.show()
