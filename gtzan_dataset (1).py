import os
import random
import librosa
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset,Subset
import soundfile as sf
from sklearn.model_selection import train_test_split


class GTZANDataset(Dataset):
    def __init__(self, audio_dir, genre_mapping, num_augmentations=4,
                 segment_duration=0.96, hop_size=1, sr=16000, start_time=0, 
                 preprocessed_dir=None, load_preprocessed=False, device='cpu'):
        """
        Dataset class for GTZAN with Mel-spectrogram processing and stored augmentations.

        Parameters:
            audio_dir (str): Directory containing audio files organized by genre.
            genre_mapping (dict): Dictionary mapping genre names to numerical labels.
            num_augmentations (int): Number of augmentations per file.
            segment_duration (float): Duration of each audio segment in seconds.
            hop_size (float): Hop size for segmenting the audio in seconds.
            sr (int): Sampling rate of the audio.
            start_time (float): Start time to begin segmenting the audio.
            preprocessed_dir (str): Directory to store/retrieve preprocessed Mel-spectrograms.
            load_preprocessed (bool): Whether to load preprocessed data if available.
            device (str): Device for data loading ('cpu' or 'cuda').
        """
        self.audio_dir = audio_dir
        self.genre_mapping = genre_mapping
        self.num_augmentations = num_augmentations
        self.segment_duration = segment_duration
        self.hop_size = hop_size
        self.sr = sr
        self.start_time = start_time
        self.preprocessed_dir = preprocessed_dir
        self.load_preprocessed = load_preprocessed
        self.device = device

        self.audio_files = []
        self.labels = []

        self._load_metadata()


    def _extract_base_id(self, file_path):
        """Extract the base ID from a file name, removing augmentation suffixes."""
        base_name = os.path.basename(file_path)
        return base_name.split('_aug')[0]

    def split_by_song_id(self, train_ratio=0.8):
        """
        Split dataset based on unique song IDs, ensuring augmentations stay in the same set.

        Parameters:
            train_ratio (float): Proportion of data for the training set.

        Returns:
            train_dataset, val_dataset: Subsets of the dataset.
        """
        # Step 1: Extract base IDs
        base_ids = [self._extract_base_id(file_path) for file_path in self.audio_files]

        # Step 2: Get unique base IDs
        unique_base_ids = list(set(base_ids))

        # Step 3: Split base IDs into train and val sets
        train_base_ids, val_base_ids = train_test_split(unique_base_ids, train_size=train_ratio, random_state=42)

        # Step 4: Assign files to train or val set based on base ID
        train_indices = [i for i, base_id in enumerate(base_ids) if base_id in train_base_ids]
        val_indices = [i for i, base_id in enumerate(base_ids) if base_id in val_base_ids]

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)

        return train_dataset, val_dataset


    def _prepare_augmented_data(self):
        """Generate and save augmented audio files in the original directory with modified names."""
        for genre, label in self.genre_mapping.items():
            genre_dir = os.path.join(self.audio_dir, genre)
            if not os.path.isdir(genre_dir):
                continue

            for file_name in os.listdir(genre_dir):
                if not file_name.endswith(('.wav', '.mp3')):
                    continue

                original_path = os.path.join(genre_dir, file_name)
                base_name = os.path.splitext(file_name)[0]

                for i in range(1, self.num_augmentations + 1):
                    augmented_path = os.path.join(genre_dir, f"{base_name}_aug{i}.wav")
                    if not os.path.exists(augmented_path):
                        y, sr = librosa.load(original_path, sr=self.sr)
                        y_augmented = self._apply_augmentation(y, sr, i)
                        sf.write(augmented_path, y_augmented, sr)

    def _apply_augmentation(self, y, sr, aug_type):
        """Apply a specific augmentation based on aug_type."""
        if aug_type == 1:
            return librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))  # Time stretching
        elif aug_type == 2:
            return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.randint(-5, 5))  # Pitch shifting
        elif aug_type == 3:
            return y + np.random.normal(0, 0.005, size=y.shape)  # Add Gaussian noise
        elif aug_type == 4:
            return np.roll(y, random.randint(1, len(y) // 10))  # Time shifting
        else:
            return y

    def _load_metadata(self):
        """Scan the directory and collect file paths and labels."""
        for genre, label in self.genre_mapping.items():
            genre_dir = os.path.join(self.audio_dir, genre)
            if not os.path.isdir(genre_dir):
                continue

            for file_name in os.listdir(genre_dir):
                if file_name.endswith(('.wav', '.mp3')):
                    self.audio_files.append(os.path.join(genre_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)

        if self.load_preprocessed and self._preprocessed_exists(audio_path):
            mel_specs = self._load_preprocessed(audio_path)
        else:
            mel_specs = self._process_audio(audio_path)
            self._save_preprocessed(audio_path, mel_specs)

        return torch.tensor(mel_specs, dtype=torch.float32).unsqueeze(1).to(self.device), label

    def _process_audio(self, audio_path):
        """Process an audio file to segment it and convert segments into Mel-spectrograms."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(self.hop_size * sr)
        start_sample = int(self.start_time * sr)

        segments = [
            y[i:i + segment_samples]
            for i in range(start_sample, len(y) - segment_samples + 1, hop_samples)
        ]

        log_mel_segments = [
            librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=segment, sr=sr, n_fft=400, hop_length=160, n_mels=64
                )
            ).T[:96, :64]
            for segment in segments
        ]

        log_mel_segments = np.array(log_mel_segments)
        return self._adjust_mel_spectrogram_length(log_mel_segments)

    def _adjust_mel_spectrogram_length(self, mel_segments):
        """Ensure the Mel-spectrogram has exactly 30 segments."""
        num_segments = mel_segments.shape[0]

        if num_segments < 30:
            last_segment = mel_segments[-1:]  # Shape: (1, 96, 64)
            repeated_segments = np.repeat(last_segment, 30 - num_segments, axis=0)
            mel_segments = np.concatenate([mel_segments, repeated_segments], axis=0)
        elif num_segments > 30:
            mel_segments = mel_segments[:30]

        return mel_segments

    def _preprocessed_exists(self, audio_path):
        """Check if preprocessed data exists for the given audio file."""
        if not self.preprocessed_dir:
            return False
        return os.path.exists(self._get_preprocessed_path(audio_path))

    def _save_preprocessed(self, audio_path, mel_specs):
        """Save preprocessed data for the given audio file."""
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        save_path = self._get_preprocessed_path(audio_path)
        with open(save_path, 'wb') as f:
            pickle.dump(mel_specs, f)

    def _load_preprocessed(self, audio_path):
        """Load preprocessed data for the given audio file."""
        load_path = self._get_preprocessed_path(audio_path)
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        return self._adjust_mel_spectrogram_length(data)

    def _get_preprocessed_path(self, audio_path):
        """Get the path for saving/loading preprocessed data."""
        base_name = os.path.basename(audio_path).replace('.mp3', '').replace('.wav', '')
        return os.path.join(self.preprocessed_dir, f"{base_name}_mel_specs.pkl")
    
# import os
# from gtzan_dataset import GTZANDataset

# def perform_augmentation(audio_dir, genre_mapping, num_augmentations=4, sr=16000):
#     """
#     Perform augmentation on the GTZAN dataset and save augmented audio files.

#     Parameters:
#         audio_dir (str): Directory containing audio files organized by genre.
#         genre_mapping (dict): Dictionary mapping genre names to numerical labels.
#         num_augmentations (int): Number of augmentations per file.
#         sr (int): Sampling rate for loading and augmenting audio files.
#     """
#     print(f"Starting augmentation for dataset in {audio_dir}...")

#     # Initialize the dataset with augmentation preparation
#     dataset = GTZANDataset(
#         audio_dir=audio_dir,
#         genre_mapping=genre_mapping,
#         num_augmentations=num_augmentations,
#         sr=sr,
#         preprocessed_dir=None,  # Not using preprocessed directory here
#         load_preprocessed=False,
#         device='cpu'
#     )

#     print("Augmentation completed successfully.")

# if __name__ == "__main__":
#     # Dataset directory and genre mapping
#     audio_dir = "gtzan-dataset-music-genre-classification/versions/1/Data/genres_original"  # Replace with your dataset directory
#     genre_mapping = {
#         "blues": 0, "classical": 1, "country": 2, "disco": 3,
#         "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
#         "reggae": 8, "rock": 9
#     }

#     # Perform augmentation
#     perform_augmentation(
#         audio_dir=audio_dir,
#         genre_mapping=genre_mapping,
#         num_augmentations=4,  # Number of augmentations per file
#         sr=16000
#     )
