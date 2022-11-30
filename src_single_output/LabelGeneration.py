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
from Dataset import TestDataset, save_label
from Train import binary_output, angle_loss, generation
from models.NewCNN import NewCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

config = {
    'batch_size' : 64,
    'architecture' : 'Xception', # change the model here
    # task and variable here means the labels to be added, not the dataset
    'task' : 'LR_task', # 'LR_task'/'Direction_task'/'Position_task' change it here
    'variable' : 'LR', # 'LR_task': 'LR'; 'Direction_task': 'Angle'/'Amplitude'; 'Position_task': 'Position'
    'synchronisation' : 'antisaccade_synchronised',#'processing_speed_synchronised',

    # I deleted configurations for processing and hilbert, use min and full data by default
    'learning_rate' : 0.0001,
}

exsiting_path = './data/'+config['task']+ '_with_' + config['synchronisation']+'_min.npz'
data_path = ['./data/LR_task_with_antisaccade_synchronised_min.npz',
        './data/Direction_task_with_dots_synchronised_min.npz',
        './data/Position_task_with_dots_synchronised_min.npz'
        ]


input_shape = (129, 500)
output_shape = 2 if config['task'] == 'Position_task' else 1 # For position tasks we have two output, but for others only one
    
if config['architecture'] == 'Xception':
    model = Xception(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

elif config['architecture'] == 'CNN':
    model = CNN(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

elif config['architecture'] == 'PyramidalCNN':
    model = PyramidalCNN(input_shape, output_shape, kernel_size=16, nb_filters=64, depth=6, batch_size=config['batch_size'])

elif config['architecture'] == 'NewCNN':
    model = CNN(input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])
    
model = model.to(device)

criterion = nn.BCEWithLogitsLoss() if config['task']=='LR_task' else nn.MSELoss()
if config['variable'] == 'Angle' and config['task']=='Direction_task':
    criterion = angle_loss
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) #Defining Optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=0.0001, verbose=True)
if config['task']=='LR_task':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=0.0001, verbose=True)
scaler = torch.cuda.amp.GradScaler()

# load models
checkpoint_path = './checkpoints/'+config['architecture']+'_'+config['task']+'_'+config['variable']+'_checkpoint.pth'

best_model = torch.load(checkpoint_path)
model.load_state_dict(best_model['model_state_dict'])
optimizer.load_state_dict(best_model['optimizer_state_dict'])

# create labels for 
for i in range(len(data_path)):
    if data_path[i] == exsiting_path:
        save_label(data_path[i], config['variable'], IsGenerated = False)
    else:
        test_data = TestDataset(data_path[i])

        test_loader = torch.utils.data.DataLoader(test_data, num_workers=2,
                                                batch_size=config['batch_size'], pin_memory=True,
                                                shuffle=False)
        pred_list = generation(model,test_loader)

        if config['task'] ==  'LR_task':
            pred_list = torch.tensor(pred_list)
            pred_list = binary_output(pred_list)
        
        generated_label = np.array(pred_list)

        save_label(data_path[i], config['variable'], IsGenerated = True, generated_label = generated_label)
