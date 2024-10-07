import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image

folder_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

image_data = []
labels = []

dataset_path = r'C:\Users\suhan\Desktop\Capstone\alzheimers_detection\ADNI DATASET'

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


image_data = np.array(image_data)
labels = np.array(labels)
print("Image data shape:", image_data.shape)
print("Labels shape:", labels.shape)

image_data = image_data.reshape(image_data.shape[0], -1)


n_folds = 5


kf = KFold(n_folds, shuffle=True, random_state=42)


accuracy_scores = []


for train_index, test_index in kf.split(image_data):
    X_train, X_test = image_data[train_index], image_data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    
    classifiers = [
        ('svm_linear', SVC(kernel='linear', probability=True)),
        ('svm_rbf', SVC(kernel='rbf', probability=True)),
        ('svm_poly', SVC(kernel='poly', probability=True))
    ]
    voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
    voting_clf.fit(X_train_pca, y_train)

    
    y_pred = voting_clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

print(f'Average Accuracy: {np.mean(accuracy_scores):.3f}')