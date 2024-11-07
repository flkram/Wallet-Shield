import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

# Custom dataset class for handling transaction data
class TransactionDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

# Feature Scaling Class (for preprocessing)
class FeatureScaler:
    def __init__(self, method='standard'):
        """
        Scales features using either StandardScaler or MinMaxScaler
        method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method. Use 'standard' or 'minmax'.")

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def transform(self, data):
        return self.scaler.transform(data)

# Advanced Fraud Detection Model with more layers and regularization
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        
        # First Layer: Fully Connected with Dropout and Batch Normalization
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        
        # Second Layer: Another fully connected layer with BatchNorm and Dropout
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        
        # Third Layer: Deeper layer with higher capacity
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.3)
        
        # Fourth Layer: A smaller layer to learn more specific patterns
        self.fc4 = nn.Linear(64, 32)
        
        # Output Layer: Single unit with sigmoid activation for binary classification
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)  # Output probability between 0 and 1 (fraud probability)
        return x

# Function to load the dataset from CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    labels = data['fraud'].values  # Assuming 'fraud' column exists (1 for fraud, 0 for valid)
    features = data.drop(['fraud'], axis=1).values  # Drop 'fraud' column to keep features
    
    # Split into train/test data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Model training function with early stopping and learning rate scheduler
def train_model(model, train_loader, criterion, optimizer, epochs=10, patience=5):
    model.train()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for batch in train_loader:
            inputs = batch['data']
            labels = batch['label']
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.float())
            
            # Calculate loss
            loss = criterion(outputs, labels.float().view(-1, 1))  # BCE Loss
            running_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()  # Threshold at 0.5 for classification
            correct_preds += (predicted.view(-1) == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        train_losses.append(epoch_loss)

        # Early stopping: Check validation loss improvement
        if epoch % 2 == 0:  # Every 2 epochs, evaluate on validation set
            val_loss, val_accuracy = evaluate_model(model, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Check if the validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_model(model)  # Save the model if it improves
            else:
                epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return train_losses, val_losses, val_accuracies

# Model evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct_preds = 0
    total_preds = 0
    running_loss = 0.0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['data']
            labels = batch['label']

            # Forward pass
            outputs = model(inputs.float())
            
            # Calculate loss
            loss = criterion(outputs, labels.float().view(-1, 1))
            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_preds += (predicted.view(-1) == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    avg_loss = running_loss / len(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    return avg_loss, accuracy

# Function to save the trained model
def save_model(model, path="fraud_detection_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the trained model
def load_model(path="fraud_detection_model.pth"):
    model = FraudDetectionModel(input_dim=10)  # Adjust input_dim based on your dataset
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

# Function to plot the training and validation loss
def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main execution to train and evaluate the model
if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data('transactions_data.csv')
    
    # Feature scaling (use 'standard' or 'minmax')
    scaler = FeatureScaler(method='standard')
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create DataLoader
    train_dataset = TransactionDataset(X_train, y_train)
    test_dataset = TransactionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Validation DataLoader (20% of train data for validation)
    val_size = int(0.2 * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
    # Validation DataLoader (20% of train data for validation)
    val_size = int(0.2 * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the model
    model = FraudDetectionModel(input_dim=X_train.shape[1])

    # Set the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Train the model
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, criterion, optimizer, epochs=25, patience=3)

    # Plot the training and validation losses
    plot_loss(train_losses, val_losses)

    # Evaluate the model on the test data
    test_loss, test_accuracy = evaluate_model(model, test_loader)

    # Display a classification report and confusion matrix for detailed evaluation
    print("Classification Report on Test Data:")
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            inputs = batch['data']
            labels = batch['label']
            outputs = model(inputs.float())
            predicted = (outputs > 0.5).float()
            y_pred.extend(predicted.view(-1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Valid', 'Fraud'])
    plt.yticks(tick_marks, ['Valid', 'Fraud'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Save the model
    save_model(model, path="final_fraud_detection_model.pth")

    # Optionally, load the model back and evaluate again (for demonstration purposes)
    print("Loading the trained model...")
    model = load_model("final_fraud_detection_model.pth")
    test_loss, test_accuracy = evaluate_model(model, test_loader)

