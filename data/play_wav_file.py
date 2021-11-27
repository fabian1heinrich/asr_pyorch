import torchaudio

from IPython.display import Audio


def play_wav_file(wav_file):

    waveform, sample_frequency = torchaudio.load(wav_file)

    return Audio(waveform, rate=sample_frequency)
