import random
from torch.utils.data import Dataset


class MixedDatasetPercentage(Dataset):
    def __init__(self, size, dataset1, dataset2, percentage1):

        self.size = size
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.percentage1 = percentage1

    def __getitem__(self, index):
        if random.random() < self.percentage1:
            return self.dataset1.__getitem__(index % len(self.dataset1))
        else:
            return self.dataset2.__getitem__(index % len(self.dataset2))

    def __len__(self):
        return self.size


class MixedDatasetComplete(Dataset):
    def __init__(self, dataset1, dataset2):

        self.dataset1 = dataset1
        self.dataset2 = dataset2

        self.length1 = len(dataset1)
        self.length2 = len(dataset2)

    def __getitem__(self, index):

        if index < self.length1:
            return self.dataset1.__getitem__(index)
        else:
            return self.dataset2.__getitem__(index - self.length1)

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)