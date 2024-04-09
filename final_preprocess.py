import mne
import numpy as np
import pandas as pd
import pywt
from mne.preprocessing import ICA
from autoreject import AutoReject


# Define item mapping for events
item_mapping = {"child": 0, "daughter": 1, "father": 2, "wife": 3, "four": 4, "three": 5, "ten": 6, "six": 7}

#Collect word orders from a text file 
wordorder = []
with open(r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-05_ses-EEG_task-inner_events.txt", "r") as x:
    for line in x:
        store = line.split()
        wordorder.append(store[2])
wordorder = wordorder[1:]  # Remove header or initial unwanted line if necessary

# Load EEG data
#raw = mne.io.read_raw_bdf(r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-01_ses-EEG_task-inner_eeg (1).bdf", preload=True)
raw = mne.io.read_raw_bdf(r"C:\Users\kapad\OneDrive\Documents\Neuro\Events and Timings\sub-05_ses-EEG_eeg.bdf", preload=True)

raw.drop_channels(['EXG7', 'EXG8'])  # Drop unwanted channels

# Update the channels of interest to include the two most important channels related to inner speech
channels_of_interest = ['F3', 'F4']
raw.pick_channels(channels_of_interest)

# Filter the data
raw.filter(l_freq=0.5, h_freq=50., method='fir', fir_window='hamming', fir_design='firwin')
raw.plot_psd()
notches = np.arange(50, 251, 50)
raw.notch_filter(notches, picks='eeg', filter_length='auto', phase='zero-double', fir_design='firwin')

# Update function name and logic for one-second segments
def create_one_second_events_with_same_labels(raw, original_duration=4.0, segment_duration=1.0, start_offset=1.0, end_offset=1.0):
    sfreq = raw.info['sfreq']
    num_samples = len(raw.times)
    original_event_duration = int(sfreq * original_duration)
    segment_event_duration = int(sfreq * segment_duration)
    start_sample_offset = int(sfreq * start_offset)
    end_sample_offset = int(sfreq * end_offset)
    events = []
    for start in range(0, num_samples, original_event_duration):
        for sub_start in range(start + start_sample_offset, start + original_event_duration - end_sample_offset, segment_event_duration):
            if start // original_event_duration < len(wordorder):
                event_label = item_mapping.get(wordorder[start // original_event_duration], 0)
                events.append([sub_start, 0, event_label])
            else:
                break
    return np.array(events)

events = create_one_second_events_with_same_labels(raw)

# Epoch the data around these new one-second segments
epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=1.0, baseline=(None, None), preload=True)

# Apply AutoReject and other processing steps as before


# Set montage for standard electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# Epoch the data around the 4-second segments
epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=4, baseline=(0, 0), preload=True)

def wavelet_denoise_epochs(epochs, wavelet='sym8', level=5):
    denoised_epochs = []
    for epoch in epochs.get_data():
        # Normalize the epoch data
        epoch_normalized = epoch / np.max(np.abs(epoch), axis=1)[:, None]
        # Apply wavelet denoising and reconstruct the signal
        denoised_epoch = []
        for data in epoch_normalized:
            # Perform discrete wavelet transform
            cA, cD = pywt.dwt(data, wavelet)
            # Apply thresholding to the detail coefficients
            cD_thresh = pywt.threshold(cD, value=0.5, mode='soft')
            # Reconstruct the signal using the thresholded detail coefficients
            data_reconstructed = pywt.idwt(cA, cD_thresh, wavelet)
            denoised_epoch.append(data_reconstructed)
        denoised_epochs.append(np.array(denoised_epoch))
    return np.array(denoised_epochs)



# Denoise the epochs
denoised_data = wavelet_denoise_epochs(epochs)


# Now, denoised_data contains your processed EEG data that you can further use for ML models or analysis

# Save the processed data for further analysis
# Adjust the path and filename as necessary
np.save(r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\np5.npy', denoised_data)

labels = events[:, 2]

# Save the labels to a file
np.save(r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\labels5.npy', labels) 



