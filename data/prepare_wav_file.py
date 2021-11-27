import torchaudio
from torchaudio import transforms

from torchaudio.transforms import Resample, MelSpectrogram
from data.collate_fn import collate_fn
from data.prepare_waveform import prepare_waveform


def prepare_wav_file(wav_file, sample_frequency_out=16000):

    waveform, sample_frequency = torchaudio.load(wav_file)

    return prepare_waveform(waveform, sample_frequency, sample_frequency_out)
