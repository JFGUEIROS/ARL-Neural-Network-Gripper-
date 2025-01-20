import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1. Custom Dataset Class
# ==========================

class SensorDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (list of np.ndarray): List of sensor data arrays.
            labels (list or np.ndarray): Corresponding list of labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the numpy array to a torch tensor
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

# ==========================
# 2. Collate Function
# ==========================

def collate_fn(batch):
    """
    Collate function to pad variable-length sequences.

    Args:
        batch (list of tuples): Each tuple is (sample, label).

    Returns:
        padded_seqs (Tensor): Padded sequences tensor of shape (max_len, batch_size, feature_dim).
        lengths (Tensor): Original lengths of each sequence.
        labels (Tensor): Labels tensor.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=False, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, labels

# ==========================
# 3. LSTM Model Definition
# ==========================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_layers=2, num_classes=2, dropout=0.5):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False  # Because DataLoader provides (max_len, batch_size, input_size)
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, padded_seqs, lengths):
        """
        Args:
            padded_seqs (Tensor): Padded sequences, shape (max_len, batch_size, input_size)
            lengths (Tensor): Original lengths of each sequence, shape (batch_size,)

        Returns:
            logits (Tensor): Output logits, shape (batch_size, num_classes)
        """
        # Pack the padded sequences
        packed_input = pack_padded_sequence(
            padded_seqs,
            lengths.cpu(),
            enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)

        # Use the final hidden state of the last LSTM layer
        final_hidden = hn[-1]  # Shape: (batch_size, hidden_size)

        # Pass through the fully connected layer
        logits = self.fc(final_hidden)  # Shape: (batch_size, num_classes)

        return logits

# ==========================
# 4. Data Loading and Preparation
# ==========================

def load_data(square_folder, ball_folder):
    """
    Loads .npy files from square and ball folders and assigns labels.

    Args:
        square_folder (Path): Path to the folder containing square .npy files.
        ball_folder (Path): Path to the folder containing ball .npy files.

    Returns:
        all_data (list of np.ndarray): Combined list of all data samples.
        all_labels (list of int): Corresponding list of labels.
    """
    def load_from_folder(folder, label):
        data = []
        labels = []
        for file in folder.glob("*.npy"):
            array = np.load(file)
            data.append(array)
            labels.append(label)
        return data, labels

    square_data, square_labels = load_from_folder(square_folder, label=1)  # Square labeled as 1
    ball_data, ball_labels = load_from_folder(ball_folder, label=0)        # Sphere labeled as 0

    all_data = square_data + ball_data
    all_labels = square_labels + ball_labels

    print(f"Total samples: {len(all_data)}")
    print(f" - Square readings: {sum(np.array(all_labels) == 1)}")
    print(f" - Sphere readings: {sum(np.array(all_labels) == 0)}")

    return all_data, all_labels

# ==========================
# 5. Plotting Functions
# ==========================

def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Plots training and testing loss and accuracy over epochs.

    Args:
        train_losses (list of float): Training loss per epoch.
        test_losses (list of float): Testing loss per epoch.
        train_accuracies (list of float): Training accuracy per epoch.
        test_accuracies (list of float): Testing accuracy per epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, test_losses, 'ro-', label='Testing Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'ro-', label='Testing Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')  # Save the figure
    print("Training history plot saved as 'training_history.png'.")
    plt.show(block=True)

def plot_confusion_matrix(cm, classes=['Sphere', 'Square']):
    """
    Plots the confusion matrix as a heatmap.

    Args:
        cm (array-like): Confusion matrix.
        classes (list): List of class names.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the figure
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")
    plt.show()

# ==========================
# 6. Main Training Pipeline
# ==========================

def main():
    # Paths to the processed data folders
    square_data_folder = Path("processed_square_readings")
    ball_data_folder = Path("processed_ball_readings")

    # Check if data folders exist
    if not square_data_folder.exists():
        print(f"Error: The folder '{square_data_folder}' does not exist.")
        return
    if not ball_data_folder.exists():
        print(f"Error: The folder '{ball_data_folder}' does not exist.")
        return

    # Load data and labels
    all_data, all_labels = load_data(square_data_folder, ball_data_folder)

    if len(all_data) == 0:
        print("Error: No data found. Please check your data folders.")
        return

    # Convert lists to NumPy arrays for compatibility
    # Note: Sequences have variable lengths; for splitting, we only need labels
    indices = np.arange(len(all_data))
    labels = np.array(all_labels)

    # Perform Stratified Train-Test Split with 70-30 ratio
    train_indices, test_indices, y_train, y_test = train_test_split(
        indices,
        labels,
        test_size=0.7,  # 60-40 split
        stratify=labels,
        random_state=42
    )

    print(f"Training samples: {len(train_indices)}")
    print(f" - Square readings: {sum(y_train == 1)}")
    print(f" - Sphere readings: {sum(y_train == 0)}")
    print(f"Testing samples: {len(test_indices)}")
    print(f" - Square readings: {sum(y_test == 1)}")
    print(f" - Sphere readings: {sum(y_test == 0)}")

    # Define the datasets
    train_data = [all_data[i] for i in train_indices]
    train_labels = y_train.tolist()

    test_data = [all_data[i] for i in test_indices]
    test_labels = y_test.tolist()

    # Create Dataset instances
    train_dataset = SensorDataset(train_data, train_labels)
    test_dataset = SensorDataset(test_data, test_labels)

    # Compute class weights to handle class imbalance
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32)
    print(f"Class Weights: {class_weights}")

    # Create DataLoaders
    BATCH_SIZE = 20

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("DataLoaders created successfully.")

    # ==========================
    # 7. Model, Loss, Optimizer Setup
    # ==========================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LSTMClassifier().to(device)

    # Initialize loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # ==========================
    # 8. Training Loop
    # ==========================

    NUM_EPOCHS = 9  # Adjust based on performance and overfitting

    # Lists to store metrics for plotting
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_loss = float('inf')
    patience = 5
    trigger_times = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (padded_seqs, lengths, labels) in enumerate(train_loader):
            padded_seqs = padded_seqs.to(device)  # (max_len, batch_size, 4)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(padded_seqs, lengths)  # (batch_size, 2)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # ==========================
        # 9. Evaluation on Test Set
        # ==========================

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for padded_seqs, lengths, labels in test_loader:
                padded_seqs = padded_seqs.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                outputs = model(padded_seqs, lengths)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total

        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        # Classification Report
        report = classification_report(all_targets, all_preds, target_names=['Sphere', 'Square'])
        cm = confusion_matrix(all_targets, all_preds)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f" - Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%")
        print(f" - Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
        print(" - Classification Report:")
        print(report)
        print(" - Confusion Matrix:")
        print(cm)
        print("-" * 50)

        # ==========================
        # 10. Early Stopping Mechanism
        # ==========================

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_lstm_classifier.pth')
            print(" - Best model saved.")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    # ==========================
    # 11. Final Model Saving
    # ==========================

    # Optionally, save the final model if it's better than previous
    torch.save(model.state_dict(), "final_lstm_classifier.pth")
    print("Final model saved to 'final_lstm_classifier.pth'.")

    # ==========================
    # 12. Plotting Training History
    # ==========================

    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies)

    # ==========================
    # 13. Plotting Confusion Matrix
    # ==========================

    plot_confusion_matrix(cm, classes=['Sphere', 'Square'])

if __name__ == "__main__":
    main()
