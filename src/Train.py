import os
import torch
import sklearn
from tqdm.auto import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, optimizer, criterion, dataloader):

    model.train()
    train_loss = 0.0  # Monitoring Loss

    for iter, (mfccs, phonemes) in enumerate(dataloader):

        # Move Data to Device (Ideally GPU)
        mfccs = mfccs.to(device)
        phonemes = phonemes.to(device)

        # Forward Propagation
        logits = model(mfccs)
        print(logits)
        # Loss Calculation
        loss = criterion(logits, phonemes)
        train_loss += loss.item()

        # Initialize Gradients
        optimizer.zero_grad()

        # Backward Propagation
        loss.backward()

        # Gradient Descent
        optimizer.step()

    train_loss /= len(dataloader)
    return train_loss


def eval(model, dataloader):

    model.eval()  # set model in evaluation mode

    phone_true_list = []
    phone_pred_list = []

    for i, data in enumerate(dataloader):

        frames, phonemes = data
        # Move data to device (ideally GPU)
        frames, phonemes = frames.to(device), phonemes.to(device)

        with torch.inference_mode():  # makes sure that there are no gradients computed as we are not training the model now
            # Forward Propagation
            logits = model(frames)

        # Get Predictions
        predicted_phonemes = torch.argmax(logits, dim=1)

        # Store Pred and True Labels
        phone_pred_list.extend(predicted_phonemes.cpu().tolist())
        phone_true_list.extend(phonemes.cpu().tolist())

        # Do you think we need loss.backward() and optimizer.step() here?

        # del frames, phonemes, logits, loss
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    # Calculate Accuracy
    accuracy = sklearn.metrics.accuracy_score(phone_pred_list, phone_true_list)
    return accuracy*100


def test(model, test_loader):
    model.eval()  # set model in evaluation mode

    # List to store predicted phonemes of test data
    test_predictions = []

    # Which mode do you need to avoid gradients?
    with torch.inference_mode():

        for i, frames in enumerate(tqdm(test_loader)):

            frames = frames.float().to(device)

            output = model(frames)

            predicted_phonemes = torch.argmax(output, dim=1)
            # How do you store predicted_phonemes with test_predictions? Hint, look at eval
            test_predictions.extend(predicted_phonemes.cpu().tolist())
    return test_predictions
