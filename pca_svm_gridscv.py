import os
import cv2  # for image processing
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define paths for each class
data_dir = r"C:\Users\suhan\Desktop\Capstone\alzheimers_detection\ADNI DATASET"  # Update with the path to your ADNI dataset
categories = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

img_size = 128  # Resize all images to 128x128 for uniformity

def load_images(data_dir, categories):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # Assign numerical label
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_img = cv2.resize(img_array, (img_size, img_size))
                data.append([resized_img, class_num])
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return data

# Apply Gabor filters to images
def apply_gabor_filters(X, img_size, ksize=5):
    gabor_features = []
    gabor_kernels = []
    
    # Generate Gabor kernels with different orientations (theta)
    for theta in range(4):  # 4 orientations (0, 45, 90, 135 degrees)
        theta = theta / 4.0 * np.pi
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    
    # Apply Gabor filter to each image in the dataset
    for image in X:
        gabor_response = np.zeros_like(image)
        for kernel in gabor_kernels:
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            gabor_response += fimg
        gabor_features.append(gabor_response.flatten())
    
    return np.array(gabor_features)

data = load_images(data_dir, categories)

# Shuffle the dataset
random.shuffle(data)

# Separate features and labels
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# Convert to numpy arrays and normalize
X = np.array(X)  # Keep as 2D images for Gabor filter processing
y = np.array(y)

# Apply Gabor filter feature extraction
X_gabor = apply_gabor_filters(X, img_size)

# Flatten original images for PCA and combine with Gabor features
X_flat = X.reshape(-1, img_size * img_size) / 255.0  # Normalize pixel values
X_combined = np.concatenate((X_flat, X_gabor), axis=1)  # Combine original and Gabor features

# Step 1: Tune PCA by plotting explained variance
pca = PCA().fit(X_combined)  # Fit PCA on the combined dataset
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()

# Based on the plot, let's say we choose n_components that explains ~95% of the variance:
pca = PCA(n_components=0.95)  # This will choose the number of components to retain 95% of variance
X_pca = pca.fit_transform(X_combined)

# Step 2: Use GridSearchCV to tune SVM parameters

# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'degree': [2, 3, 4]  # Degree for 'poly' kernel
}

# Initialize the SVM model
svm = SVC()

# Set up grid search with 5-fold cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2)

# Fit the grid search model on PCA-transformed data
grid_search.fit(X_pca, y)

# Best parameters found by grid search
print("Best parameters found by GridSearchCV:", grid_search.best_params_)

# Use the best estimator (SVM model with the best parameters)
best_svm = grid_search.best_estimator_

# Step 3: Evaluate the best model with k-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X_pca):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the best SVM model
    best_svm.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Output the cross-validated accuracy
print(f"Cross-validated accuracy of the best SVM model: {np.mean(accuracies)}")

# Optional Step 4: Visualize SVM decision boundary (with PCA-reduced data to 2D for visualization)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_combined)

# Train the best SVM on 2D data
best_svm.fit(X_2d, y)

# Create a meshgrid to plot decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundary
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)

# Plot data points
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors='k', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary with PCA-reduced Data')
plt.show()