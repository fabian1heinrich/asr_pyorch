import torch.nn as nn
import torchaudio

from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

from model import train_model, train_model_timing, ASRModel2, save_model
from data import collate_fn

train_dataset = torchaudio.datasets.LIBRISPEECH(
    "./",
    url="train-clean-100",
    download=True,
)
test_dataset = torchaudio.datasets.LIBRISPEECH(
    "./",
    url="test-clean",
    download=True,
)
training_transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
)

test_transform = MelSpectrogram(sample_rate=16000, n_mels=128)

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: collate_fn(x, training_transform),
    drop_last=False,
)

model = ASRModel2()
train_model(model, train_loader, 30)
save_model(model)

print("debug")
