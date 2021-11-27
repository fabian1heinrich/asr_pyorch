from IPython.display import Audio


def play_waveform(waveform, sample_frequency):

    return Audio(waveform, rate=sample_frequency)
