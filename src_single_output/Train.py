import os
import torch
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def angle_loss(a, b):
    return torch.mean(torch.square(torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))))

def binary_output(x):
    x = torch.sigmoid(x)
    output = np.zeros(x.shape)
    output[x >= 0.3] = 1
    output = torch.tensor(output)
    return output

def get_output(pred, true, task, variable):
    pred = torch.tensor(pred)
    true = torch.tensor(true)
    # pred = pred * (label_max - label_min) + label_min
    # true = true*(label_max-label_min) + label_min
    if task ==  'LR_task':
        pred = binary_output(pred)

        measure = accuracy_score(true,pred)
        print("\tAccuracy: {:.2f}%".format(measure*100))
    elif task == 'Direction_task':
        if variable == 'Angle':
            pred = pred.numpy()
            true = true.numpy()
            measure = np.sqrt(np.mean(np.square(np.arctan2(np.sin(true - pred.ravel()), np.cos(true - pred.ravel())))))
            print("\tAngle mean squared error: {:.4f}".format(measure))
        elif variable == 'Amplitude':
            measure = mean_squared_error(pred,true,squared=False)
            measure = measure/2 # pixel -> mm
            print("\tAmplitude mean squared error: {:.4f}".format(measure))
    elif task == 'Position_task':
        measure = np.linalg.norm(true - pred, axis=1).mean()
        measure = measure/2 # pixel -> mm
        print("\tEuclidean distance: {:.4f}".format(measure))
    return measure, pred

def train(model, optimizer, criterion, scaler, dataloader):

    model.train()
    train_loss = 0.0  # Monitoring Loss

    phone_true_list = []
    phone_pred_list = []

    # Progress Bar 
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    

    for iter, (mfccs, phonemes) in enumerate(dataloader):
        
        # Move Data to Device (Ideally GPU)
        mfccs = mfccs.to(device)
        phonemes = phonemes.to(device)
        
        phonemes = torch.squeeze(phonemes)

        with torch.cuda.amp.autocast(): # This implements mixed precision. Thats it! 
            # Forward Propagation
            logits = model(mfccs)
            logits = logits.to(torch.float64)
            phonemes = phonemes.to(torch.float64)
            logits = torch.squeeze(logits)
            # Loss Calculation
            loss = criterion(logits, phonemes)
            phone_pred_list.extend(logits.cpu().tolist())
            phone_true_list.extend(phonemes.cpu().tolist())

        
        # Update no. of correct predictions & loss as we iterate
        train_loss += loss.item()

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(train_loss / (iter + 1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        # Initialize Gradients
        optimizer.zero_grad()

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() 

        batch_bar.update() # Update tqdm bar

    batch_bar.close() # You need this to close the tqdm bar


    train_loss /= len(dataloader)
    return train_loss, phone_pred_list, phone_true_list


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


        # Store Pred and True Labels
        phone_pred_list.extend(logits.cpu().tolist())
        phone_true_list.extend(phonemes.cpu().tolist())

        # Do you think we need loss.backward() and optimizer.step() here?

        # del frames, phonemes, logits, loss
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    # Calculate Accuracy
    
    return phone_pred_list, phone_true_list


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


        # Store Pred and True Labels
        phone_pred_list.extend(logits.cpu().tolist())
        phone_true_list.extend(phonemes.cpu().tolist())

        # Do you think we need loss.backward() and optimizer.step() here?

        # del frames, phonemes, logits, loss
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    # Calculate Accuracy
    return phone_pred_list, phone_true_list

def generation(model,test_loader):
    model.eval()  # set model in evaluation mode

    pred_list = []

    for i, data in enumerate(test_loader):

        # Move data to device (ideally GPU)
        data = data.to(device)

        with torch.inference_mode():  # makes sure that there are no gradients computed as we are not training the model now
            # Forward Propagation
            predicted_label = model(data)


        # Store Pred Labels
        pred_list.extend(predicted_label.cpu().tolist())

        del data, predicted_label
        torch.cuda.empty_cache()

    # Calculate Accuracy
    return pred_list
