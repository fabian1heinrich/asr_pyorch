import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

from data import collate_fn, MixedDatasetComplete
from model import ASRModel2, train_model, save_model
from airbus_dataset import AirbusDataset

model = ASRModel2()

airbus_dataset = AirbusDataset()
libri_dataset = torchaudio.datasets.LIBRISPEECH(
    "./",
    url="train-clean-100",
    download=True,
)
training_transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
)

dataset = MixedDatasetComplete(airbus_dataset, libri_dataset)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=20,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    collate_fn=lambda x: collate_fn(x, training_transform),
)

train_model(model, data_loader, 30)
save_model(model)
print("debug")
