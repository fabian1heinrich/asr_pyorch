import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence

from data.text_to_target import text_to_target


def collate_fn(data_batch, transform):

    inputs = []
    targets = []

    # waveform, sample_frequency, utterance, speaker_id, chapter_id, utterance_id
    for waveform, sample_frequency, utterance, _, _, _ in data_batch:
        inputs.append(transform(waveform.squeeze()).transpose_(0, 1))
        targets.append(text_to_target(utterance))

    inputs, input_lengths = pad_packed_sequence(
        pack_sequence(inputs, enforce_sorted=False),
        batch_first=True,
    )
    inputs = inputs.unsqueeze(1)
    # gru bidirectional
    input_lengths = torch.div(input_lengths, 2, rounding_mode='floor')

    targets, target_lengths = pad_packed_sequence(
        pack_sequence(targets, enforce_sorted=False),
        batch_first=True,
    )
    return inputs, targets, input_lengths, target_lengths
    # inputs BxCxTxF
