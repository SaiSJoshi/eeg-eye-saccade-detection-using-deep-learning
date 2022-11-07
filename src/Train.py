import os
import torch
import sklearn
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, optimizer, criterion, dataloader):

    model.train()
    train_loss = 0.0  # Monitoring Loss
    num_correct = 0

    # Progress Bar 
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 

    for iter, (mfccs, phonemes) in enumerate(dataloader):

        # Move Data to Device (Ideally GPU)
        mfccs = mfccs.to(device)
        phonemes = phonemes.to(device)

        # Forward Propagation
        logits = model(mfccs)
        phonemes = torch.flatten(phonemes)
        # Loss Calculation
        loss = criterion(logits, phonemes)
       
       # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(logits, axis=1) == phonemes).sum())
        train_loss += loss.item()

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (64*(iter + 1))),
            loss="{:.04f}".format(float(train_loss / (iter + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        # Initialize Gradients
        optimizer.zero_grad()

        # Backward Propagation
        loss.backward()

        # Gradient Descent
        optimizer.step()

        batch_bar.update() # Update tqdm bar

    batch_bar.close() # You need this to close the tqdm bar

    acc = 100 * num_correct / (64* len(dataloader))

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
    accuracy = accuracy_score(phone_pred_list, phone_true_list)
    return accuracy*100


# def test(model, test_loader):
#     model.eval()  # set model in evaluation mode

#     # List to store predicted phonemes of test data
#     test_predictions = []

#     # Which mode do you need to avoid gradients?
#     with torch.inference_mode():

#         for i, frames in enumerate(tqdm(test_loader)):

#             frames = frames.float().to(device)

#             output = model(frames)

#             predicted_phonemes = torch.argmax(output, dim=1)
#             # How do you store predicted_phonemes with test_predictions? Hint, look at eval
#             test_predictions.extend(predicted_phonemes.cpu().tolist())
#     return test_predictions

def test(model,test_loader):
    model.eval()  # set model in evaluation mode

    phone_true_list = []
    phone_pred_list = []

    for i, data in enumerate(test_loader):

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
    accuracy = accuracy_score(phone_pred_list, phone_true_list)
    return accuracy*100, phone_pred_list, phone_true_list