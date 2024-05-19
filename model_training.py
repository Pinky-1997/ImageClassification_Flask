
import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Define paths
dataset_path = 'dataset_full'
categories = ['Building', 'Forest', 'Glacier', 'Mountain', 'Seas', 'Street']

# Initialize data and labels
data = []
labels = []

# Preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at path: {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))  # Resize to a fixed size
    equalized = cv2.equalizeHist(resized)  # Histogram equalization
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)  # Gaussian smoothing
    return blurred

# Load and preprocess images
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        processed_img = preprocess_image(img_path)
        if processed_img is not None:
            data.append(processed_img.flatten())
            labels.append(categories.index(category))

data = np.array(data)
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train_pca, y_train)

# Save model
with open('model.pkl', 'wb') as model_file:
    pickle.dump((scaler, pca, classifier), model_file)

# Evaluate model
y_pred = classifier.predict(X_test_pca)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
