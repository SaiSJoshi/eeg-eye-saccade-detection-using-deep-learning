import numpy as np
import torch
import torch.nn as nn
import os
from Dataset import Dataset
from models.MyXception import Xception
from models.MyPyramidalCNN import PyramidalCNN
from models.MyCNN import CNN
from Train import train, eval, test


def main():

    config = {
        'epochs': 25,
        'batch_size' : 1,
        'learning_rate' : 0.001,
        'architecture' : 'Xception', # change the model here
        'task' : 'LR_task', # 'LR_task'/'Direction_task'/'Position_task' change it here
        'synchronisation' : 'antisaccade_synchronised',
        'hilbert' : True, # with (True) or without (False) hilbert transform
        'preprocessing' : 'min', # min/max
        'train_ratio' : 0.7,
        'val_ratio' : 0.15,
        'test_ratio' : 0.15
        
    }


    data_path = 'data/'+config['task']+ '_with_' + config['synchronisation']+'_'+config['preprocessing']
    data_path = data_path+'_hilbert.npz' if config['hilbert'] else data_path+'.npz'

    train_data = Dataset(data_path, train_ratio = config['train_ratio'], val_ratio = config['val_ratio'], test_ratio = config['test_ratio'], partition = 'train')
    val_data = Dataset(data_path, train_ratio = config['train_ratio'], val_ratio = config['val_ratio'], test_ratio = config['test_ratio'], partition = 'val')
    test_data = Dataset(data_path, train_ratio = config['train_ratio'], val_ratio = config['val_ratio'], test_ratio = config['test_ratio'], partition = 'test')


    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4,
                                            batch_size=config['batch_size'], pin_memory=True,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data, num_workers=2,
                                            batch_size=config['batch_size'], pin_memory=True,
                                            shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data, num_workers=2,
                                            batch_size=config['batch_size'], pin_memory=True,
                                            shuffle=False)

    # Testing code to check if my data loaders are working
    for i, data in enumerate(train_loader):
        frames, phoneme = data
        print(frames.shape, phoneme.shape)
        # print(y)
        break

    # add models here

    input_shape = (1, 258) if config['hilbert'] else (500, 129)
    output_shape = 2 # only for LE tasks, need to change for other tasks
    if config['architecture'] == 'Xception':
        model = Xception(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

    elif config['architecture'] == 'CNN':
        model = Xception(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

    elif config['architecture'] == 'PyramidalCNN':
        model = Xception(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])
    frames,phoneme = next(iter(train_loader))

    criterion = nn.CrossEntropyLoss() #Defining Loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) #Defining Optimizer

    # may add wandb part later
    torch.cuda.empty_cache()

    epochs = 25
    best_acc = 0.0 ### Monitor best accuracy in your run

    for epoch in range(config['epochs']):
        print("\nEpoch {}/{}".format(epoch+1, epochs))

        train_loss = train(model, optimizer, criterion, train_loader)
        accuracy = eval(model, val_loader)

        print("\tTrain Loss: {:.4f}".format(train_loss))
        print("\tValidation Accuracy: {:.2f}%".format(accuracy))


        ### Log metrics at each epoch in your run - Optionally, you can log at each batch inside train/eval functions (explore wandb documentation/wandb recitation)
        # wandb.log({"train loss": train_loss, "validation accuracy": accuracy})

        ### Save checkpoint if accuracy is better than your current best
        if accuracy >= best_acc:

        ### Save checkpoint with information you want
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'acc': accuracy}, 
            './model_checkpoint.pth')

        ### Save checkpoint in wandb
        #   wandb.save('checkpoint.pth')

        # Is your training time very high? Look into mixed precision training if your GPU (Tesla T4, V100, etc) can make use of it 
        # Refer - https://pytorch.org/docs/stable/notes/amp_examples.html

    ### Finish your wandb run
    # run.finish()

    predictions = test(model, test_loader)

if __name__=='__main__':
    main()
