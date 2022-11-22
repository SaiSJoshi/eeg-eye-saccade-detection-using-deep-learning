import numpy as np
import torch
import torch.nn as nn
import os
import sklearn
from sklearn.metrics import accuracy_score
from torchsummary import summary

from models.MyXception import Xception
from models.MyPyramidalCNN import PyramidalCNN
from models.MyCNN import CNN
from Dataset import Dataset
from Train import train, eval, test, get_output, angle_loss
from models.NewCNN import NewCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def main():

    config = {
        'epochs': 50,
        'batch_size' : 64,
        'learning_rate' : 0.0001,
        'architecture' : 'CNN', # change the model here
        'task' : 'Direction_task', # 'LR_task'/'Direction_task'/'Position_task' change it here
        'variable' : 'Amplitude', # 'LR_task': 'LR'; 'Direction_task': 'Angle'/'Amplitude'; 'Position_task': 'Position'
        'synchronisation' : 'dots_synchronised', #'dots_synchronised',#'processing_speed_synchronised',
        'hilbert' : False, # with (True) or without (False) hilbert transform
        'preprocessing' : 'min', # min/max
        'train_ratio' : 0.7,
        'val_ratio' : 0.15,
        'test_ratio' : 0.15
        
    }


    data_path = '../data/'+config['task']+ '_with_' + config['synchronisation']+'_'+config['preprocessing']
    data_path = data_path+'_hilbert.npz' if config['hilbert'] else data_path+'.npz'

    train_data = Dataset(data_path, hilbert = config['hilbert'], train_ratio = config['train_ratio'], val_ratio = config['val_ratio'], test_ratio = config['test_ratio'], task = config['task'], variable = config['variable'], partition = 'train')
    val_data = Dataset(data_path, hilbert = config['hilbert'], train_ratio = config['train_ratio'], val_ratio = config['val_ratio'], test_ratio = config['test_ratio'], task = config['task'], variable = config['variable'], partition = 'val')
    test_data = Dataset(data_path, hilbert = config['hilbert'], train_ratio = config['train_ratio'], val_ratio = config['val_ratio'], test_ratio = config['test_ratio'], task = config['task'], variable = config['variable'], partition = 'test')


    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4,
                                            batch_size=config['batch_size'], pin_memory=True,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data, num_workers=2,
                                            batch_size=config['batch_size'], pin_memory=True,
                                            shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data, num_workers=2,
                                            batch_size=config['batch_size'], pin_memory=True,
                                            shuffle=False)

    input_shape = (1, 258) if config['hilbert'] else (129, 500)
    output_shape = 2 if config['task'] == 'Position_task' else 1 # For position tasks we have two output, but for others only one
    if config['architecture'] == 'Xception':
        model = Xception(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

    elif config['architecture'] == 'CNN':
        model = CNN(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

    elif config['architecture'] == 'PyramidalCNN':
        model = PyramidalCNN(input_shape, output_shape, kernel_size=16, nb_filters=64, depth=6, batch_size=config['batch_size'])

    elif config['architecture'] == 'NewCNN':
        model = CNN(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])
        

    frames,phoneme = next(iter(train_loader))
    model = model.to(device)
    summary(model,input_shape)

    # criterion = nn.BCEWithLogitsLoss() if config['task']=='LR_task' else nn.MSELoss()
    # if config['variable'] == 'Angle' and config['task']=='Direction_task':
    #     criterion = angle_loss

    # Losses
    lr_criterion = nn.BCEWithLogitsLoss()
    angle_criterion = nn.MSELoss()
    amplitude_criterion = nn.MSELoss()
    abs_pos_coriterion = nn.MSELoss()
    criterion = [lr_criterion, angle_criterion, amplitude_criterion, abs_pos_coriterion]
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) #Defining Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=0.0001, verbose=True)
    if config['task']=='LR_task':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=0.0001, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    # may add wandb part later
    torch.cuda.empty_cache()

    epochs = config['epochs']
    best_acc = 0.0 ### Monitor best accuracy in your run

    # Initializing for early stopping 
    best_val_meansure = 0.0
    patience_count = 0 
    patience_max = 20 # TODO: initialized based on paper
    for epoch in range(config['epochs']):
        print("\nEpoch {}/{}".format(epoch+1, epochs))

        train_loss, train_pred, train_true = train(model, optimizer, criterion, scaler, train_loader)
        print("\tTrain Loss: {:.4f}".format(train_loss))
        print("\tTrain:")
        train_measure, train_pred = get_output(train_pred, train_true, config['task'],config['variable'])
        val_pred, val_true = eval(model, val_loader)
        print("\tValidation:")
        val_measure, val_pred = get_output(val_pred, val_true, config['task'],config['variable'])
        
        ## Early Stopping condition
        if abs(val_measure - best_val_meansure) > 0.1:
            best_val_meansure = val_measure
        else: 
            patience_count += 1

        if patience_count  >= patience_max:
            print("\nValid Accuracy didn't improve since last {} epochs.", patience_count)
            break 



        ### Log metrics at each epoch in your run - Optionally, you can log at each batch inside train/eval functions (explore wandb documentation/wandb recitation)
        # wandb.log({"train loss": train_loss, "validation accuracy": accuracy})

        ### Save checkpoint if accuracy is better than your current best
        
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'acc': val_measure}, 
        '../checkpoints/'+config['architecture']+'_'+config['task']+'_checkpoint.pth')

        
        scheduler.step(val_measure)
    #     ## Save checkpoint in wandb
        #    wandb.save('checkpoint.pth')

    #     Is your training time very high? Look into mixed precision training if your GPU (Tesla T4, V100, etc) can make use of it 
    #     Refer - https://pytorch.org/docs/stable/notes/amp_examples.html

    # ## Finish your wandb run
    # run.finish()

    test_pred, test_true = test(model, test_loader)
    print("\tTest:")
    test_measure, test_pred = get_output(test_pred, test_true, config['task'],config['variable'])
    results_name = '../results/'+config['architecture']+'_'+config['task']+'_'+config['variable']+".npz"
    print(results_name)
    np.savez(results_name, pred = test_pred, truth = test_true, measure = test_measure)

if __name__=='__main__':
    main()