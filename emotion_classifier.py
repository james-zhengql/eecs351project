import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader

class EmotionClassification(nn.Module):
    def __init__(self, vggish_model):
        super(EmotionClassification, self).__init__()

        # Assign the pretrained VGGish model
        self.vggish = vggish_model

        # Freeze the VGGish model
        for param in self.vggish.parameters():
            param.requires_grad = False

        # CNN for aggregating temporal information from segment embeddings
        self.cnn_aggregator = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Aggregate over the sequence dimension
        )

        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(32, 64),  # Input: Aggregated embedding (32 -> 64)
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: Valence and Arousal (64 -> 2)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x (torch.Tensor): Input tensor in the form (batch, num_segments, 1, 96, 64).
        
        Returns:
            torch.Tensor: Predicted valence and arousal for each input batch (batch, 2).
        """
        # Get the batch size and number of segments
        batch_size, num_segments, channels, height, width = x.shape

        # Step 1: Flatten batch and segment dimensions
        # (batch, num_segments, 1, 96, 64) -> (batch * num_segments, 1, 96, 64)
        x = x.view(batch_size * num_segments, channels, height, width)

        # Step 2: Pass the input through VGGish to get segment embeddings
        with torch.no_grad():  # Ensure VGGish remains frozen
            vggish_output = self.vggish(x)  # Shape: (batch * num_segments, 128)

        # Step 3: Reshape back to original batch and segment structure
        # (batch * num_segments, 128) -> (batch, num_segments, 128)
        vggish_output = vggish_output.view(batch_size, num_segments, -1)
        vggish_output = vggish_output/255
        # Step 4: Transpose for Conv1d: (batch, num_segments, 128) -> (batch, 128, num_segments)
        vggish_output = vggish_output.permute(0, 2, 1)

        # Step 5: Aggregate segment embeddings using CNN
        aggregated_embedding = self.cnn_aggregator(vggish_output)  # Shape: (batch, 32, 1)
        aggregated_embedding = aggregated_embedding.squeeze(-1)  # Shape: (batch, 32)

        # Step 6: Predict valence and arousal using the fully connected layers
        return self.fc(aggregated_embedding)


def train_model(model, dataset, num_epochs=20, batch_size=16, lr=0.001, device='cuda'):
    """
    Train the emotion_classification model with optional fine-tuning of VGGish.
    """
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Train all parameters

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0

        for batch_audio, batch_labels in dataloader:
            # Move data to the appropriate device
            batch_audio = batch_audio.to(device)  # Shape: (batch, num_segments, 1, 96, 64)
            batch_labels = batch_labels.to(device)  # Shape: (batch, 2)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_audio)  # Shape: (batch, 2)
            loss = criterion(predictions, batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Logging the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    print("Training completed.")


