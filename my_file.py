import torch.nn as nn

from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

from data import collate_fn
from model import ASRModel2, train_model, save_model
from airbus_dataset import AirbusDataset

airbus_dataset = AirbusDataset()

training_transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=128),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
)

data_loader = DataLoader(
    dataset=airbus_dataset,
    batch_size=20,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    collate_fn=lambda x: collate_fn(x, training_transform),
)

model = ASRModel2()

train_model(model, data_loader, 20)
save_model(model)
print("debug")