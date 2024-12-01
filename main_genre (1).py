import torch
from torch.utils.data import random_split, DataLoader
from gtzan_dataset import GTZANDataset
from genre_classification import GenreClassification
from contrastive_loss_util import *


def train_genre_model_with_eval(model, train_loader, val_loader, num_epochs=20, batch_size=16, lr=0.001, device='cuda', patience=5):
    """
    Train the model with evaluation and early stopping, and print loss and accuracy.

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
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_mel_specs, batch_labels in train_loader:
            batch_mel_specs = batch_mel_specs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_mel_specs)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            correct_train += (predicted == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train

        # Evaluation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_mel_specs, batch_labels in val_loader:
                batch_mel_specs = batch_mel_specs.to(device)
                batch_labels = batch_labels.to(device)

                predictions = model(batch_mel_specs)
                loss = criterion(predictions, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                correct_val += (predicted == batch_labels).sum().item()
                total_val += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_genre_model_{model.aggregation_method}_contrastive.pth")  # Save the best model
            print("Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training completed for method '{model.aggregation_method}'. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # Dataset parameters
    audio_dir = "gtzan-dataset-music-genre-classification/versions/1/Data/genres_original"
    genre_mapping = {
        "blues": 0, "classical": 1, "country": 2, "disco": 3,
        "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
        "reggae": 8, "rock": 9
    }

    # Initialize dataset with augmented files
    dataset = GTZANDataset(
        audio_dir=audio_dir,
        genre_mapping=genre_mapping,
        preprocessed_dir="preprocessed_gtzan",
        load_preprocessed=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    train_dataset, val_dataset = dataset.split_by_song_id(train_ratio=0.8)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load pretrained VGGish model
    vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish_model.eval()
    vggish_model.preprocess = False

    # Loop through different aggregation methods
    aggregation_methods = ["transformer"]
    for method in aggregation_methods:
        print(f"Training model with aggregation method: {method}")
        
        # Initialize the genre classification model with the current aggregation method
        model = GenreClassification(vggish_model, num_classes=10, aggregation_method=method)
        
        # Train and evaluate the model
        train_genre_model_with_eval(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            batch_size=64,
            lr=0.0001,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            patience=5
        )

# if __name__ == "__main__":

#     # Dataset parameters
#     audio_dir = "gtzan-dataset-music-genre-classification/versions/1/Data/genres_original"
#     genre_mapping = {
#         "blues": 0, "classical": 1, "country": 2, "disco": 3,
#         "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
#         "reggae": 8, "rock": 9
#     }

#     # Initialize dataset with augmented files
#     dataset = GTZANDataset(
#         audio_dir=audio_dir,
#         genre_mapping=genre_mapping,
#         preprocessed_dir="preprocessed_gtzan",
#         load_preprocessed=True,
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )
#     # Split dataset by song ID
#     train_dataset, val_dataset = dataset.split_by_song_id(train_ratio=0.8)

#     # Create PairDataset for contrastive loss
#     pair_train_dataset = PairDataset(dataset, train_dataset.indices, dataset.labels)
#     pair_val_dataset = PairDataset(dataset, val_dataset.indices, dataset.labels)

#     # DataLoaders
#     pair_train_loader = DataLoader(pair_train_dataset, batch_size=16, shuffle=True)
#     pair_val_loader = DataLoader(pair_val_dataset, batch_size=16, shuffle=False)
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

#     # Load pretrained VGGish model
#     vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
#     vggish_model.eval()
#     vggish_model.preprocess = False

#     # Initialize GenreClassification model
#     model = GenreClassification(vggish_model, num_classes=10, aggregation_method="transformer")

#     # Step 1: Pretrain with Contrastive Loss
#     print("Pretraining with Contrastive Loss...")
#     pretrain_with_contrastive_loss(model, pair_train_loader, num_epochs=10, lr=0.0001, margin=1.0, device='cuda' if torch.cuda.is_available() else 'cpu')

#     # Step 2: Fine-tune with Cross-Entropy Loss
#     print("Fine-tuning with Cross-Entropy Loss...")
#     train_genre_model_with_eval(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=40,
#         batch_size=16,
#         lr=0.0001,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         patience=5
#     )
