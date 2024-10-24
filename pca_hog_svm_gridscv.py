import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skimage.feature import hog

data_dir = r"C:\Users\suhan\Desktop\Capstone\alzheimers_detection\ADNI DATASET"  # Update with the path to your ADNI dataset
categories = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

img_size = 128  

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


def apply_hog_features(X, img_size):
    hog_features = []
    for image in X:        
        hog_feature, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                                     cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        hog_features.append(hog_feature)    
    return np.array(hog_features)

data = load_images(data_dir, categories)
random.shuffle(data)
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)
X = np.array(X) 
y = np.array(y)
X_hog = apply_hog_features(X, img_size)
X_hog = X_hog / np.linalg.norm(X_hog, axis=1, keepdims=True)

#Tune PCA by plotting explained variance
pca = PCA().fit(X_hog) 
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()

pca = PCA(n_components=0.95)  
X_pca = pca.fit_transform(X_hog)

#Use GridSearchCV to tune SVM parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_pca, y)
print("Best parameters found by GridSearchCV:", grid_search.best_params_)
best_svm = grid_search.best_estimator_

# Evaluate the best model with k-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X_pca):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]
    best_svm.fit(X_train, y_train)
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)


print(f"Cross-validated accuracy of the best SVM model: {np.mean(accuracies)}")

#Visualize SVM decision boundary (with PCA-reduced data to 2D for visualization)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_hog)
best_svm.fit(X_2d, y)
h = .02  
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors='k', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary with PCA-reduced Data')
plt.show()
