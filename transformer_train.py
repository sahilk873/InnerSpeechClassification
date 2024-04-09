import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import mne
import os
import math

# Path to the .fif file
fif_files = [r'C:\Users\kapad\OneDrive\Documents\Neuro\subject1_cleaned.fif', r'C:\Users\kapad\OneDrive\Documents\Neuro\subject2_cleaned.fif', r'C:\Users\kapad\OneDrive\Documents\Neuro\subject3_cleaned.fif', r'C:\Users\kapad\OneDrive\Documents\Neuro\subject5_cleaned.fif']

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
                X = X.transpose(1, 0)  # Transpose the data matrix to match PyTorch format
                new_label = 0 if labels[i] <= 3 else 1
                self.labels.append(torch.tensor(new_label, dtype=torch.long))
                self.data.append(torch.tensor(X, dtype=torch.float32))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create dataset and dataloader
dataset = EEGDataset(fif_files)
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dim_model, num_heads, num_layers, dropout_rate=0.5):
        super(TransformerClassifier, self).__init__()
        
        # CNN layer(s)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Embedding layer to match transformer dimension, if still needed
        self.embedding = nn.Linear(dim_model, dim_model)
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(dim_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final fully connected layer
        self.fc = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        # Apply CNN layer(s)
        x = x.transpose(1, 2)  # Adjust input dimensions (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten the output for the embedding layer
        x = x.transpose(1, 2)  # Switch back to (batch_size, sequence_length, features) for transformer
        
        # Apply embedding if still needed
        x = self.embedding(x)
        
        # Apply positional encoding and transformer
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        
        # Take the output of the last sequence for classification
        out = transformer_output[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# Hyperparameters
input_size = 8  # Number of features (EEG channels)
dim_model = 64  # Dimension of the transformer model
num_heads = 4  # Number of heads in the multi-head attention models
num_layers = 3  # Number of transformer layers
num_classes = 2  # Number of output classes (event IDs)

model = TransformerClassifier(input_size, num_classes, dim_model, num_heads, num_layers).to(device)


# Train the model
num_epochs = len(dataset)  # Number of epochs
best_accuracy = 0  # Track the best accuracy
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

num_epochs = 20000  # Set a fixed number of epochs
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
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
            X, y = X.to(device), y.to(device)
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





