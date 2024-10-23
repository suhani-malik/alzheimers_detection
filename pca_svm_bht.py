import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from PIL import Image

# Define the folder names for the dataset
folder_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

image_data = []
labels = []

# Set the path to your dataset
dataset_path = r'C:\Users\suhan\Desktop\Capstone\alzheimers_detection\ADNI DATASET'

# Load the image data and labels
for folder_name in folder_names:
    folder_path = os.path.join(dataset_path, folder_name)
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    for image_file in image_files:
        image = np.array(Image.open(image_file))
        image_data.append(image)
        
        if folder_name == 'Mild_Demented':
            labels.append(0)
        elif folder_name == 'Moderate_Demented':
            labels.append(1)
        elif folder_name == 'Non_Demented':
            labels.append(2)
        elif folder_name == 'Very_Mild_Demented':
            labels.append(3)

# Convert lists to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)
print("Image data shape:", image_data.shape)
print("Labels shape:", labels.shape)

# Reshape the image data
image_data = image_data.reshape(image_data.shape[0], -1)

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=42)

accuracy_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(image_data):
    X_train, X_test = image_data[train_index], image_data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Apply ANOVA for feature selection
    anova = SelectKBest(f_classif, k=100)
    anova.fit(X_train_pca, y_train)
    X_train_anova = anova.transform(X_train_pca)
    X_test_anova = anova.transform(X_test_pca)

    # Define the SVM model
    svm = SVC(probability=True)

    # Define the parameter space for Bayesian optimization
    param_space = {
        'C': (1e-6, 1e+6, 'log-uniform'),  # Regularization parameter
        'gamma': (1e-6, 1e+1, 'log-uniform'),  # Kernel coefficient
        'kernel': ['linear', 'rbf', 'poly']  # Kernel type
    }

    # Create a Bayesian search for hyperparameter optimization
    bayes_search = BayesSearchCV(svm, param_space, n_iter=3, cv=3, n_jobs=-1, random_state=42)
    bayes_search.fit(X_train_anova, y_train)

    # Make predictions and evaluate accuracy
    y_pred = bayes_search.predict(X_test_anova)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Print average accuracy
print(f'Average Accuracy: {np.mean(accuracy_scores):.3f}')