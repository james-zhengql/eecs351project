import torch
from deam_dataset import *
from emotion_classifier import *
from torch.utils.data import random_split, DataLoader
import numpy as np

def train_model_with_eval(model, train_loader, val_loader, num_epochs=20, batch_size=16, lr=0.001, device='cuda', patience=5):
    """
    Train the model with evaluation and early stopping.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        device (str): Device to train on ('cuda' or 'cpu').
        patience (int): Number of epochs to wait for improvement before stopping early.
    """
    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_mel_specs, batch_labels in train_loader:
            batch_mel_specs = batch_mel_specs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_mel_specs)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_mel_specs, batch_labels in val_loader:
                batch_mel_specs = batch_mel_specs.to(device)
                batch_labels = batch_labels.to(device)

                predictions = model(batch_mel_specs)
                loss = criterion(predictions, batch_labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Training completed.")

if __name__ == "__main__":
    # Load pretrained VGGish model
    vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish_model.eval()
    vggish_model.preprocess = False  # Set preprocess to False

    # Initialize the emotion classification model
    model = EmotionClassification(vggish_model)

    # Dataset paths
        
    combined_csv = "filtered_annotations.csv"
    audio_dir = "DEAM_audio/MEMD_audio"
    preprocessed_dir = "preprocessed_spectrogram"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the dataset with preprocessing enabled
    dataset = AudioDataset(combined_csv, audio_dir, preprocessed_dir=preprocessed_dir, device=device, load_preprocessed=True)
    
    # Split dataset into training and evaluation sets (80% training, 20% evaluation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Train and evaluate the model
    train_model_with_eval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        batch_size=16,
        lr=0.0001,
        device=device,
        patience=3  # Early stopping patience
    )
