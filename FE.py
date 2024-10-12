import os
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

IMAGE_SIZE = (128, 128)
dataset_path = '/content/drive/MyDrive/ADNI DATASET'

def load_images(dataset_path):
    data = []
    labels = []
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)  
            data.append(img)
            labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

images, labels = load_images(dataset_path)

images = images / 255.0

# 1. Feature Extraction Using Pre-trained VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
vgg_features = vgg_model.predict(images)
vgg_features_flat = vgg_features.reshape((vgg_features.shape[0], -1))

print(f'VGG16 Extracted Features Shape: {vgg_features_flat.shape}')

# 2. Handcrafted Feature Extraction: HOG Features
def extract_hog_features(images):
    hog_features_list = []
    for img in images:
        img = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features, _ = hog(img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
        hog_features_list.append(hog_features)
    return np.array(hog_features_list)

hog_features = extract_hog_features(images)
print(f'HOG Extracted Features Shape: {hog_features.shape}')

# 3. Dimensionality Reduction Using PCA
scaler = StandardScaler()
vgg_features_scaled = scaler.fit_transform(vgg_features_flat)
hog_features_scaled = scaler.fit_transform(hog_features)

pca_vgg = PCA(n_components=2)
vgg_pca_2d = pca_vgg.fit_transform(vgg_features_scaled)

pca_hog = PCA(n_components=2)
hog_pca_2d = pca_hog.fit_transform(hog_features_scaled)

print(f'PCA-Reduced VGG Features Shape (2D): {vgg_pca_2d.shape}')
print(f'PCA-Reduced HOG Features Shape (2D): {hog_pca_2d.shape}')

# 4. t-SNE Dimensionality Reduction
tsne_vgg = TSNE(n_components=2, perplexity=30, n_iter=300)
vgg_tsne_2d = tsne_vgg.fit_transform(vgg_features_scaled)

tsne_hog = TSNE(n_components=2, perplexity=30, n_iter=300)
hog_tsne_2d = tsne_hog.fit_transform(hog_features_scaled)

print(f't-SNE Reduced VGG Features Shape (2D): {vgg_tsne_2d.shape}')
print(f't-SNE Reduced HOG Features Shape (2D): {hog_tsne_2d.shape}')

# 5. Plotting the Results
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

def plot_2d(features, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.show()

plot_2d(vgg_pca_2d, labels, 'PCA of VGG16 Features')
plot_2d(hog_pca_2d, labels, 'PCA of HOG Features')

plot_2d(vgg_tsne_2d, labels, 't-SNE of VGG16 Features')
plot_2d(hog_tsne_2d, labels, 't-SNE of HOG Features')
