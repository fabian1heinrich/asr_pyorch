import time
from datetime import datetime
import torch

from model.calc_accuracy import calc_accuracy


def train_model(model, train_loader, epochs):

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=5e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        anneal_strategy="linear",
    )
    criterion = torch.nn.CTCLoss(blank=0).to(0)
    model.train()

    timestamp = datetime.now().strftime("%Y%m%d_%I_%M_%S_%p")
    print(">> training started @{} <<".format(timestamp))

    elapsed_time = time.time()
    for epoch in range(epochs):

        train_loss = 0

        for batch in train_loader:
            inputs, targets, input_lengths, target_lengths = batch

            inputs = inputs.to(0)
            targets = targets.to(0)
            input_lengths = input_lengths.to(0)
            target_lengths = target_lengths.to(0)

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
            scheduler.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        accuracy = calc_accuracy(
            outputs,
            targets,
            input_lengths,
            target_lengths,
        )
        print(
            "epoch #{}/{} | loss: {:.4f} | accuracy: {:.2f} | elapsed time: {:.0f} s"
            .format(
                epoch + 1,
                epochs,
                train_loss,
                accuracy,
                time.time() - elapsed_time,
            ))

    timestamp = datetime.now().strftime("%Y%m%d_%I_%M_%S_%p")
    print(">> training finished @{} <<".format(timestamp))
