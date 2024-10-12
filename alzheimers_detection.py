import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from hyperopt import hp, fmin, tpe, Trials
from sklearn.metrics import precision_score, recall_score, f1_score

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
precision_scores = []
recall_scores = []
f1_scores = []


for train_index, test_index in kf.split(image_data):
    X_train, X_test = image_data[train_index], image_data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    
    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train_pca)
    X_test_tsne = tsne.fit_transform(X_test_pca)

        
    space = {
        'C': hp.loguniform('C', -5, 2),
        'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly']),
        'gamma': hp.loguniform('gamma', -5, 0),
        'degree': hp.quniform('degree', 2, 10, 1)
    }

    def objective(params):
        
        svc = SVC(**params)
        svc.degree = int(svc.degree)
        svc.fit(X_train_tsne, y_train)
        
        
        y_pred = svc.predict(X_test_tsne)
        accuracy = accuracy_score(y_test, y_pred)
        
        
        return -accuracy

    
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=50)


    svc = SVC(**best)
    svc.fit(X_train_tsne, y_train)

    
    y_pred = svc.predict(X_test_tsne)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print(f'Average Accuracy: {np.mean(accuracy_scores):.3f}')
print(f'Average Precision: {np.mean(precision_scores):.3f}')
print(f'Average Recall: {np.mean(recall_scores):.3f}')
print(f'Average F1 Score: {np.mean(f1_scores):.3f}')