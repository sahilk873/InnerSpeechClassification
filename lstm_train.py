import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import mne
import os
import numpy as np
from torchsummary import summary
import torch
from torch.utils.data import Dataset



data_files = [
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\np1.npy', 
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\np2.npy', 
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\np3.npy', 
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\np5.npy'
]

label_files = [
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\labels1.npy', 
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\labels2.npy', 
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\labels3.npy', 
    r'C:\Users\kapad\OneDrive\Documents\Neuro\Data\labels5.npy'
]

class EEGDataset(Dataset):
    """EEG dataset loaded from .npy files."""

    def __init__(self, data_files, label_files):
        
        self.data = []
        self.labels = []

        # Load data and labels from .npy files
        for data_file, label_file in zip(data_files, label_files):
            data = np.load(data_file)
            labels = np.load(label_file)

            # Assuming data shape is (epochs, channels, time points)
            # Transpose data to match PyTorch format: (epochs, time points, channels)
            data = data.transpose(0, 2, 1)

            self.data.append(torch.tensor(data, dtype=torch.float32))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

        # Concatenate all data and labels from different subjects into a single dataset
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



# Create dataset and dataloader
dataset = EEGDataset(data_files, label_files)
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# Define the LSTM model
class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_channels, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(CNNLSTMClassifier, self).__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Adjust CNN layers
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Apply CNN layers
        x = x.permute(0, 2, 1)  # Reshape x to (batch_size, num_channels, sequence_length) for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # Reshape x to (batch_size, sequence_length, features) for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Apply LSTM layers
        x, _ = self.lstm(x, (h0, c0))
        x = self.dropout(x)

        # Apply the fully connected layer
        x = self.fc(x[:, -1, :])  # Only take the output of the last LSTM cell
        return x

    
    
# Adjust the hyperparameters accordingly
num_channels = 2  # Number of EEG channels in your input data
hidden_size = 128  # Number of features in the hidden state
num_layers = 2  # Number of stacked LSTM layers
num_classes = 8  # Number of output classes (unique labels in your dataset)

model = CNNLSTMClassifier(num_channels=num_channels, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout_rate=0.5)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = len(dataset)  # Number of epochs
    
num_epochs = 200  # Set a fixed number of epochs
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == y).sum().item()
        total_predictions += y.size(0)

    # Validation accuracy
    model.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    train_accuracy = 100 * correct_predictions / total_predictions
    val_accuracy = 100 * val_correct / val_total

    print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), r"C:\Users\kapad\Downloads\lstm1224.pth")
        print("Saved model with improved validation accuracy: {:.2f}%".format(val_accuracy))





