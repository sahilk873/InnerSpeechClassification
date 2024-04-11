import mne
import numpy as np
import pandas as pd
import umap
from scipy.special import legendre, chebyt
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def reduce_dimensions_and_train_ann(epochs, labels):
    # Flatten the data for dimensionality reduction
    data = epochs.get_data().reshape(len(epochs), -1)
    
    # Dimensionality reduction using t-SNE to 8 dimensions
    tsne = TSNE(n_components=3, random_state=42)
    data_reduced = tsne.fit_transform(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_reduced, labels, test_size=0.2, random_state=42)
    
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Artificial Neural Network (ANN) for classification
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(np.unique(labels)), activation='softmax')  # Output layer; number of neurons equals the number of classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Example usage
epochs, labels = make_epochs()  # Ensure you have the make_epochs function from earlier
reduce_dimensions_and_train_ann(epochs, labels)



