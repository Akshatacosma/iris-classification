# Iris Classification

This project demonstrates the classification and visualization of decision boundaries
using multiple machine learning models on the Iris dataset with scikit-learn.

---

## Project Overview

The objective of this project is to:
- Train different classifiers on the Iris dataset
- Compare how each model classifies data
- Visualize probability surfaces and decision boundaries
- Evaluate model performance using accuracy

Only the first two features of the dataset are used to enable clear visualization.

---

## Models Used

- Logistic Regression  
- Gaussian Process Classifier  
- Gradient Boosting Classifier  

Each model classifies the data into three classes:
- Class 0
- Class 1
- Class 2

---

## Visualization Details

- Blue color intensity represents the probability of a specific class.
  - Darker shades indicate higher confidence.
  - Lighter shades indicate lower confidence.

- The "Max Class" plot displays the final predicted class at each point,
  based on the highest probability among all classes.

---

## Evaluation Metric

- Accuracy is used to compare model performance on the test dataset.

---

## Technologies Used

- Python  
- NumPy  
- Matplotlib  
- scikit-learn  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Akshatacosma/iris-classification.git
