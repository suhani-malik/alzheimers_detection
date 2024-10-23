#!pip install scikit-optimize
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Step 1: Load your 6400 image dataset from directories using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
image_size = (128, 128)

# Assuming sub-folders for categories: mild_demented, moderate_demented, non_demented, very_mild_demented
dataset = datagen.flow_from_directory(
    '/content/drive/MyDrive/ADNI DATASET',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Step 2: Extract VGG16 features
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
features = vgg_model.predict(dataset)

# Flatten the features to 1D array
vgg16_features = features.reshape(features.shape[0], -1)

# Labels for the dataset
labels = dataset.classes

# Step 3: Split Data into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(vgg16_features, labels, test_size=0.3, random_state=42)

# Step 4: Apply PCA to reduce dimensionality (use 2 components for visualization)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 5: Apply t-SNE to reduce dimensionality (use 2 components for visualization)
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test)

# Step 6: Combine PCA and t-SNE features
X_train_combined = np.hstack((X_train_pca, X_train_tsne))
X_test_combined = np.hstack((X_test_pca, X_test_tsne))

# Normalize the combined features
scaler = StandardScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_test_combined = scaler.transform(X_test_combined)

# Step 7: SVM with Bayesian Hyperparameter Tuning
search_spaces = {
    'C': (1e-6, 1e+6, 'log-uniform'),  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types
    'gamma': (1e-6, 1e+1, 'log-uniform')  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
}

# Set up the Bayesian optimization for SVM
opt = BayesSearchCV(
    SVC(),
    search_spaces,
    n_iter=10,  # Number of iterations for Bayesian optimization
    cv=3,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available cores
    random_state=42
)

# Step 8: Fit the model to the training data
opt.fit(X_train_combined, y_train)

# Step 9: Predict on the test data
y_pred = opt.predict(X_test_combined)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
