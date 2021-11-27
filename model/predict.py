import torch
from data import target_to_text


def predict(outputs, input_lengths):

    prediction = []

    _, max_index = torch.max(outputs, dim=2)

    for i, cur_seq in enumerate(max_index):

        pred = cur_seq[0:input_lengths[i]]
        pred = torch.unique_consecutive(pred)
        pred = pred[pred.nonzero(as_tuple=True)]

        prediction.append(pred)

    return prediction