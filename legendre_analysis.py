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


item_mapping = {"child": 0, "daughter": 1, "father": 2, "wife": 3, "four": 4, "three": 5, "ten": 6, "six": 7}


# Dictionary of EEG files and their corresponding event files
files = {
    r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-01_ses-EEG_task-inner_eeg (1).bdf": r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-01_ses-EEG_task-inner_events.txt",
    r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-05_ses-EEG_eeg.bdf": r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-05_ses-EEG_task-inner_events.txt",
    r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-02_ses-EEG_eeg.bdf": r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-02_ses-EEG_task-inner_events.txt",
    r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-03_ses-EEG_eeg.bdf": r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-03_ses-EEG_task-inner_events.txt",
}



# Function to process each EEG file
def process_eeg(eeg_file_path, events_file_path):
    # Load EEG data
    raw = mne.io.read_raw_bdf(eeg_file_path, preload=True)
    raw.drop_channels(['EXG7', 'EXG8'])  # Drop unwanted channels
    channels_of_interest = ['F3']
    raw.pick_channels(channels_of_interest)
    raw.filter(l_freq=0.5, h_freq=50., method='fir', fir_window='hamming', fir_design='firwin')
    # Notch filter to remove line noise
    notches = np.arange(50, 251, 50)
    raw.notch_filter(notches, picks='eeg', filter_length='auto', phase='zero-double', fir_design='firwin')

    # Collect word orders from the events file
    wordorder = []
    with open(events_file_path, "r") as x:
        for line in x:
            store = line.split()
            wordorder.append(store[2])
    wordorder = wordorder[1:]  # Remove header or initial unwanted line if necessary

    # Create and process events
    events = create_middle_two_second_events(raw, wordorder)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=2.0, baseline=(None, None), preload=True)
    return epochs  # or any other data you wish to return or save

# Function to create events for the middle two seconds of each four-second segment, adjusted to use word order
def create_middle_two_second_events(raw, wordorder, duration=4.0):
    sfreq = raw.info['sfreq']
    num_samples = len(raw.times)
    event_duration_samples = int(sfreq * duration)
    middle_offset_samples = int(sfreq * (duration / 2.0 - 1))

    events = []
    for start in range(0, num_samples, event_duration_samples):
        if start // event_duration_samples < len(wordorder):
            middle_start = start + middle_offset_samples
            if middle_start < num_samples:
                event_id = item_mapping.get(wordorder[start // event_duration_samples], 0)
                events.append([middle_start, 0, event_id])
        else:
            break
    return np.array(events)



def make_epochs():
    
    all_epochs_list = []
    all_labels = []
# Iterate over the files dictionary and process each EEG file
    for eeg_path, events_path in files.items():
        epochs = process_eeg(eeg_path, events_path)
        wordorder = []
        with open(events_path, "r") as x:
            for line in x:
                if line.strip():  # skip empty lines
                    store = line.split()
                    wordorder.append(store[2])
        wordorder = wordorder[1:]  # Skip header or initial unwanted line if necessary
        
        # Append the epochs from this file to the list
        all_epochs_list.append(epochs)
        
        # Create labels for these epochs based on the wordorder
        file_labels = [item_mapping[word] for word in wordorder]
        all_labels.extend(file_labels)

# Concatenate all epochs into a single Epochs object
    all_epochs = mne.concatenate_epochs(all_epochs_list)
    return all_epochs, all_labels
    
  
def analyze_basic(epochs, labels):
    
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

    order = 10  # Degree of the polynomial
    n_epochs, n_channels, n_times = data.shape

    # Decompose signals into Legendre polynomials
    coefficients = np.zeros((n_epochs, n_channels, order + 1))
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            signal = data[epoch, channel, :]
            for deg in range(order + 1):
                poly = legendre(deg)
                coefficients[epoch, channel, deg] = np.dot(poly(np.linspace(-1, 1, n_times)), signal)

    # Flatten coefficients for feature selection
    coefficients_flat = coefficients.reshape(n_epochs, -1)
    
    variances = np.var(coefficients_flat, axis=0)

    # Variance-based feature selection
    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    variances = np.var(coefficients_flat, axis=0)

    # Determine the number of features to keep, e.g., top 20%
    top_percentage = 0.25  # Keep top 20% of features
    num_features_to_keep = int(len(variances) * top_percentage)

    # Sort the variances and select the indices of the top features
    indices_of_top_features = np.argsort(variances)[-num_features_to_keep:]
    print(indices_of_top_features)
    # Select only the top features based on calculated indices
    coefficients_selected = coefficients_flat[:, indices_of_top_features]
    
    #Lasso Based Feature
    #lasso = LassoCV(cv=5).fit(coefficients_flat, labels)
    #importance = np.abs(lasso.coef_)
    #coefficients_selected = coefficients_flat[:, importance > 0.01]  
    
    

    #  Applying LDA for dimensionality reduction
    lda = LDA(n_components=3)  # We aim for a 3D feature space
    #X_lda = lda.fit_transform(coefficients_selected, labels)
    
    
    # Example placeholder code for PCA followed by LDA
    
    pca = PCA(n_components=12)  # Adjust based on dataset specifics
    coefficients_pca = pca.fit_transform(coefficients_selected)
    X_lda = lda.fit_transform(coefficients_pca, labels)

    # Applying PCA for dimensionality reduction (Uncomment to use)
    # pca = PCA(n_components=3)
    # X_transformed = pca.fit_transform(coefficients_selected)

    # Applying Kernel PCA for dimensionality reduction (Uncomment to use)
    #kpca = KernelPCA(n_components=3, kernel='rbf', gamma=15)
    #X_transformed = kpca.fit_transform(coefficients_selected)
    #X_lda = X_transformed

    # Visualization in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming 'labels' is a numpy array with the same length as the rows in X_lda
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
    unique_labels = np.unique(labels)  # Unique classes in your labels

    # Iterate over the unique labels and plot each in a different color
    for i, label in enumerate(unique_labels):
        # Select the data points that belong to the current label
        ix = np.where(labels == label)
        ax.scatter(X_lda[ix, 0], X_lda[ix, 1], X_lda[ix, 2], c=colors[i % len(colors)], label=f'Class {label}', s=50)

    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    plt.title('LDA: 3D Feature Space for 8 Classes')
    plt.legend()

    # Remove manual axis limits to let matplotlib auto-adjust
    # ax.set_xlim([X_lda[:,0].min(), X_lda[:,0].max()])
    # ax.set_ylim([X_lda[:,1].min(), X_lda[:,1].max()])
    # ax.set_zlim([X_lda[:,2].min(), X_lda[:,2].max()])

    plt.show()
