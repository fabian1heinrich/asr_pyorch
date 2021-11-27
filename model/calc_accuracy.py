from model.predict import predict
from data import target_to_text


def calc_accuracy(outputs, targets, input_lengths, target_lengths):

    accuracy = 0
    counter = 0

    prediction = predict(outputs, input_lengths)

    # targets = targets.to("cpu")

    for i, pred in enumerate(prediction):

        pred = pred.to(targets.device)
        targ = targets[i]
        accuracy += sum(cur_p == cur_t for cur_p, cur_t in zip(pred, targ))
        counter += target_lengths[i]

    print(target_to_text(prediction[i]))
    print(target_to_text(targets[i]))
    return accuracy / counter
