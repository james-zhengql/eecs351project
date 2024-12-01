import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

class GenreClassification(nn.Module):
    def __init__(self, vggish_model, num_classes=10, aggregation_method="cnn"):
        super(GenreClassification, self).__init__()

        self.aggregation_method = aggregation_method

        # Assign the pretrained VGGish model
        self.vggish = vggish_model

        # Freeze the VGGish model
        # for param in self.vggish.parameters():
        #     param.requires_grad = False

        # CNN for aggregating temporal information
        self.cnn_aggregator = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Aggregate over the sequence dimension
        )

        # Transformer for aggregating temporal information
        if aggregation_method == "transformer":
            # self.pos_encoder = SinusoidalPositionalEncoding(d_model=128, max_len=30)
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
            # self.fc_sequence = nn.Linear(30, 1)  # Input: seq_len=30, Output: 1

        # self.normalization = nn.LayerNorm(128)

        # Fully connected layers for genre classification
        self.fc_input_size = 64 if aggregation_method == "cnn" else 128

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),  # Adjust input size dynamically
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output: Genre classes
        )

    # def forward(self, x):
    #     """
    #     Forward pass.

    #     Parameters:
    #         x (torch.Tensor): Input tensor in the form (batch, num_segments, 1, 96, 64).
        
    #     Returns:
    #         torch.Tensor: Predicted genre logits for each input batch (batch, num_classes).
    #     """
    #     # Get the batch size and number of segments
    #     batch_size, num_segments, channels, height, width = x.shape

    #     # Step 1: Flatten batch and segment dimensions
    #     x = x.view(batch_size * num_segments, channels, height, width)

    #     # Step 2: Pass the input through VGGish to get segment embeddings
    #     with torch.no_grad():
    #         vggish_output = self.vggish(x)  # Shape: (batch * num_segments, 128)

    #     # Step 3: Reshape back to original batch and segment structure
    #     vggish_output = vggish_output.view(batch_size, num_segments, -1)
    #     vggish_output = vggish_output / 255.0  # Normalize embeddings

    #     # Step 4: Transpose for Conv1d: (batch, num_segments, 128) -> (batch, 128, num_segments)
    #     vggish_output = vggish_output.permute(0, 2, 1)

    #     # Step 5: Apply aggregation method
    #     if self.aggregation_method == "cnn":
    #         aggregated_embedding = self.cnn_aggregator(vggish_output).squeeze(-1)  # Shape: (batch, 64)
    #     elif self.aggregation_method == "transformer":
    #         # Transformer expects input as (seq_length, batch_size, embedding_dim)
    #         vggish_output = vggish_output.permute(2, 0, 1)  # Shape: (num_segments, batch, 128)
    #         # Apply positional encoding
    #         # vggish_output = self.pos_encoder(vggish_output)  # Add positional encoding
    #         transformer_output = self.transformer(vggish_output)  # Shape: (num_segments, batch, 128)
    #         # Transpose for Fully Connected Layer
    #         # transformer_output = transformer_output.permute(1, 2, 0)  # Shape: (16, 128, 30)
    #         aggregated_embedding = transformer_output.mean(dim=0)  # Aggregate over sequence dimension
    #         # aggregated_embedding = self.fc_sequence(transformer_output).squeeze(-1)
    #     elif self.aggregation_method == "mean":
    #         aggregated_embedding = vggish_output.mean(dim=-1)  # Aggregate using mean pooling
    #     elif self.aggregation_method == "max":
    #         aggregated_embedding = vggish_output.max(dim=-1).values  # Aggregate using max pooling
    #     else:
    #         raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
    #     normalized_embedding = self.normalization(aggregated_embedding)
    #     # Step 6: Predict genres using the fully connected layers
    #     return self.fc(normalized_embedding)
    
    # for contrastive loss
    def forward(self, x, return_embedding=False):
        # Preprocess and extract VGGish embeddings
        batch_size, num_segments, channels, height, width = x.shape
        x = x.view(batch_size * num_segments, channels, height, width)
        with torch.no_grad():
            vggish_output = self.vggish(x)

        vggish_output = vggish_output.view(batch_size, num_segments, -1) / 255.0
        vggish_output = vggish_output.permute(1, 0, 2)
        if self.aggregation_method == "transformer":
            transformer_output = self.transformer(vggish_output).mean(dim=0)
        else:
            raise ValueError("Only transformer aggregation is supported for contrastive loss.")

        if return_embedding:
            return transformer_output  # Return embeddings for contrastive loss

        # normalized_embedding = self.normalization(transformer_output)
        return self.fc(transformer_output)  # Return logits for classification


def train_genre_model(model, dataset, num_epochs=20, batch_size=16, lr=0.001, device='cuda'):
    """
    Train the GenreClassification model with optional fine-tuning of VGGish.
    """
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Train all parameters

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0

        for batch_audio, batch_labels in dataloader:
            # Move data to the appropriate device
            batch_audio = batch_audio.to(device)  # Shape: (batch, num_segments, 1, 96, 64)
            batch_labels = batch_labels.to(device)  # Shape: (batch)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_audio)  # Shape: (batch, num_classes)
            loss = criterion(predictions, batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Logging the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    print("Training completed.")
