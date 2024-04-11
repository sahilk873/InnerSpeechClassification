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
    
    
    
def analyze_tsne(epochs, labels):
    data = epochs.get_data().reshape(len(epochs), -1)  # Reshaping data
    
    # Mutual Information based feature selection
    mi = mutual_info_classif(data, labels)
    k_best = SelectKBest(score_func=lambda X, Y: mi, k='all')
    data_mi_selected = k_best.fit_transform(data, labels)
    
    # PCA with whitening
    pca = PCA(n_components=0.95, whiten=True)  # Adjust n_components as needed
    data_pca_whitened = pca.fit_transform(data_mi_selected)
    
    # t-SNE for dimensionality reduction to 3D
    X_tsne = TSNE(n_components=3, perplexity=40, random_state=42).fit_transform(data_pca_whitened)
    
    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=labels, cmap='tab10')
    
    ax.set_xlabel('TSNE-1')
    ax.set_ylabel('TSNE-2')
    ax.set_zlabel('TSNE-3')
    plt.title('t-SNE visualization of EEG data')
    plt.show()

    



