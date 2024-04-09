import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mne
import os

# Path to the .fif file
fif_files = [r'C:\Users\kapad\OneDrive\Documents\Neuro\subject1.fif', r'C:\Users\kapad\OneDrive\Documents\Neuro\subject2.fif', r'C:\Users\kapad\OneDrive\Documents\Neuro\subject3.fif', r'C:\Users\kapad\OneDrive\Documents\Neuro\subject5.fif']

class EEGDataset(Dataset):
    """EEG dataset."""

    def __init__(self, fif_file):
        self.data = []
        self.labels = []

        for fif_file in fif_files:
            epochs = mne.read_epochs(fif_file)
            labels = epochs.events[:, 2]

            for i in range(len(epochs)):
                X = epochs[i].get_data().squeeze(0)
                X = X.transpose(0, 1)  # Transpose the data matrix to match PyTorch format
                self.data.append(torch.tensor(X, dtype=torch.float32))
                self.labels.append(torch.tensor(labels[i], dtype=torch.long))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create dataset and dataloader
dataset = EEGDataset(fif_files)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 2049  # Number of features (EEG channels)
hidden_size = 64  # Number of features in hidden state
num_layers = 2  # Number of stacked LSTM layers
num_classes = 8  # Number of output classes (event IDs)

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)


if os.path.exists(r'C:\Users\kapad\Downloads\model.pth'):
    model.load_state_dict(torch.load(r'C:\Users\kapad\Downloads\model.pth'))
    print("Loaded weights from the best model.")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = len(dataset)  # Number of epochs
best_accuracy = 0  # Track the best accuracy
        
        
num_epochs = 2000  # Set a fixed number of epochs
best_accuracy = 0  # Track the best accuracy

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += y.size(0)
        correct_predictions += (predicted == y).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100 * correct_predictions / total_predictions
    print(f'Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        torch.save(model.state_dict(), r'C:\Users\kapad\Downloads\model.pth')
        print("Saved model with accuracy: {:.2f}%".format(epoch_accuracy))





