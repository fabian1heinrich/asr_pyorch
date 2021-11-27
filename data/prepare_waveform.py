import torchaudio
from torchaudio import transforms

from torchaudio.transforms import Resample, MelSpectrogram
from data.collate_fn import collate_fn


def prepare_waveform(waveform, sample_frequency, sample_frequency_out=16000):

    transform = MelSpectrogram(sample_rate=16000, n_mels=128)

    resample = Resample(
        orig_freq=sample_frequency,
        new_freq=sample_frequency_out,
    )

    waveform = resample(waveform)
    data_batch = [waveform, sample_frequency, 'WAV FILE', 0, 0, 0]
    input, _, input_length, _ = collate_fn([data_batch], transform)

    return input, input_length
