# Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Program
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    # Load Iris dataset
    iris = datasets.load_iris()
    
    # Select first two features
    X = iris.data[:, :2]
    y = iris.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Linear SVM
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X_train, y_train)
    
    # Linear predictions
    y_pred_linear = linear_svm.predict(X_test)
    print("Linear SVM Accuracy:",
          accuracy_score(y_test, y_pred_linear))
    
    # Train RBF SVM
    rbf_svm = SVC(kernel='rbf')
    rbf_svm.fit(X_train, y_train)
    
    # RBF predictions
    y_pred_rbf = rbf_svm.predict(X_test)
    print("RBF SVM Accuracy:",
          accuracy_score(y_test, y_pred_rbf))
    
    # Plot decision boundaries
    def plot_boundary(model, title):
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )
    
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_train[:, 0], X_train[:, 1],
                    c=y_train, edgecolors='black')
    
        plt.title(title)
        plt.xlabel("Sepal Length (Standardized)")
        plt.ylabel("Sepal Width (Standardized)")
        plt.show()
    
    plot_boundary(linear_svm, "Linear SVM Decision Boundary")
    plot_boundary(rbf_svm, "RBF SVM Decision Boundary")


## output
     Linear SVM Accuracy: 0.7333333333333333
    RBF SVM Accuracy: 0.7333333333333333



## Result
Thus, the Support Vector Machine (SVM) classification model was successfully implemented using Python. Linear and RBF kernels were trained and compared using the Iris dataset. Their classification performance was evaluated using accuracy, and decision boundaries were visualized.
