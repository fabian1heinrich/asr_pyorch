import torch

alphabet_dict = {
    "'": 1,
    " ": 2,
    "A": 3,
    "B": 4,
    "C": 5,
    "D": 6,
    "E": 7,
    "F": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "J": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "O": 17,
    "P": 18,
    "Q": 19,
    "R": 20,
    "S": 21,
    "T": 22,
    "U": 23,
    "V": 24,
    "W": 25,
    "X": 26,
    "Y": 27,
    "Z": 28
}

index_dict = dict((v, k) for k, v in alphabet_dict.items())


def text_to_target(text, dict=alphabet_dict):
    target = []

    for letter in text:
        target.append(dict[letter])

    return torch.Tensor(target)


def target_to_text(target, dict=index_dict):
    text = []
    text_str = ""

    for index in target[target.nonzero(as_tuple=True)]:
        text.append(dict[int(index.item())])

    return text_str.join(text)
