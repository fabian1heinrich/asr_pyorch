import time
import torch


def train_model_timing(model, train_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CTCLoss(blank=0).to(0)
    model.train()

    batch_wait, model_train = [], []
    t1, t2, t3 = 0, 0, 0

    print(">> training started <<")
    for batch in train_loader:
        inputs, targets, input_lengths, target_lengths = batch

        inputs = inputs.to(0)
        targets = targets.to(0)
        input_lengths = input_lengths.to(0)
        target_lengths = target_lengths.to(0)
        t1 = time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(
            outputs.permute(1, 0, 2),
            targets,
            input_lengths,
            target_lengths,
        )
        loss.backward()
        optimizer.step()

        t2 = time.time()
        batch_wait.append(t1 - t3)
        model_train.append(t2 - t1)
        t3 = time.time()

    batch_prep_mean = sum(batch_wait[1:]) / (len(batch_wait) - 1) * 10**3
    model_train_mean = sum(model_train) / len(model_train) * 10**3
    print("batch_wait: {:.0f} ms | model_train: {:.0f} ms".format(
        batch_prep_mean, model_train_mean))
    print(">> training finished <<")
    return batch_prep_mean, model_train_mean
