import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import os
import pickle


class AudioDataset(Dataset):
    def __init__(self, combined_csv, audio_dir, preprocessed_dir=None, 
                 segment_duration=0.96, hop_size=1, sr=16000, start_time=15, device='cpu', load_preprocessed=False):
        self.device = device
        self.preprocessed_dir = preprocessed_dir
        self.labels_data = pd.read_csv(combined_csv)
        self.labels = self.labels_data[[' valence_mean', ' arousal_mean']].values
        self.song_ids = self.labels_data['song_id'].tolist()
        self.audio_dir = audio_dir
        self.segment_duration = segment_duration
        self.hop_size = hop_size
        self.sr = sr
        self.start_time = start_time
        self.load_preprocessed = load_preprocessed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get song ID and labels
        song_id = int(self.song_ids[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)

        if self.load_preprocessed and self._preprocessed_exists(song_id):
            # Load preprocessed mel-spectrograms and labels
            mel_specs,_ = self._load_preprocessed(song_id)
        else:
            # Process audio file and save the mel-spectrograms
            audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")
            mel_specs = self._process_audio(audio_path, self.segment_duration, self.hop_size, self.sr, self.start_time)
            if self.preprocessed_dir:
                self._save_preprocessed(song_id, mel_specs, labels)
        # Return mel-spectrograms and labels
        return torch.tensor(mel_specs, dtype=torch.float32).unsqueeze(1).to(self.device), labels

    def _process_audio(self, audio_path, segment_duration, hop_size, sr, start_time):
        """
        Process an audio file to segment it and convert segments into mel-spectrograms.

        Returns:
            np.ndarray: A 3D array of mel-spectrograms, shape (num_segments, 96, 64).
        """
        y, sr = librosa.load(audio_path, sr=sr)
        segment_samples = int(segment_duration * sr)
        hop_samples = int(hop_size * sr)
        start_sample = int(start_time * sr)
        segments = [
            y[i:i + segment_samples]
            for i in range(start_sample, len(y) - segment_samples + 1, hop_samples)
        ]

        log_mel_segments = []
        for segment in segments:
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=400, hop_length=160, n_mels=64)
            log_mel_spec = librosa.power_to_db(mel_spec)
            log_mel_segments.append(log_mel_spec.T[:96, :64])

        assert len(log_mel_segments) == 30, f"Audio file {audio_path} does not have 30 segments. Found {len(log_mel_segments)} segments."

        return np.array(log_mel_segments)

    def _preprocessed_exists(self, song_id):
        """Check if preprocessed data exists for the given song."""
        if not self.preprocessed_dir:
            return False
        return os.path.exists(os.path.join(self.preprocessed_dir, f"{song_id}_mel_specs.pkl"))

    def _save_preprocessed(self, song_id, mel_specs, labels):
        """Save preprocessed data for the given song."""
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        save_path = os.path.join(self.preprocessed_dir, f"{song_id}_mel_specs.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({'mel_specs': mel_specs, 'labels': labels.cpu().numpy()}, f)

    def _load_preprocessed(self, song_id):
        """Load preprocessed data for the given song."""
        load_path = os.path.join(self.preprocessed_dir, f"{song_id}_mel_specs.pkl")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        mel_specs = data['mel_specs']
        labels = torch.tensor(data['labels'], dtype=torch.float32).to(self.device)
        return mel_specs, labels


# Example Usage
# combined_csv = "static_annotations_averaged_songs_1_2000.csv"
# audio_dir = "DEAM_audio/MEMD_audio"
# preprocessed_dir = "preprocessed_spectrogram"

# dataset = AudioDataset(combined_csv, audio_dir, preprocessed_dir=preprocessed_dir, load_preprocessed=True)
# mel_spec, label = dataset[0]
# print("Mel-spectrogram shape:", mel_spec.shape)
# print("Label (valence, arousal):", label)
