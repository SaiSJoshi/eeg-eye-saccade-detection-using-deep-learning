import numpy as np
import torch
import torch.nn as nn
import os
import sklearn
from sklearn.metrics import accuracy_score
from torchsummary import summary

from models.MyXception import Xception
from models.MyPyramidalCNN import PyramidalCNN
# from models.MyCNN import CNN
from models.multitask_CNN import CNN
from multitask_Dataset import Dataset
from multitask_Train import train, eval, test, get_output, angle_loss
from models.NewCNN import NewCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


config = {
    'epochs': 50,
    'batch_size' : 16,
    'learning_rate' : 0.0001,
    'architecture' : 'CNN', # change the model here 
}
weight_LR = [0.25, 0.25]
weight_angle = [0.25, 0.25]
weight_amp = [0.25, 0.25]
weight_pos = [0.25, 0.25]

weights =[weight_LR,weight_angle,weight_amp,weight_pos] # 4*2 the first value for the generated data, the second value for the original data


# TODO: import as list for datapath
train_datapath = "../data/Generated_train.npz"
val_datapath = "../data/Generated_val.npz"
test_datapath = "../data/Generated_test.npz"

torch.cuda.empty_cache()

train_data = Dataset(train_datapath)
val_data = Dataset(train_datapath)
test_data = Dataset(train_datapath)


train_loader = torch.utils.data.DataLoader(train_data, num_workers=4,
                                        batch_size=config['batch_size'], pin_memory=True,
                                        shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data, num_workers=2,
                                        batch_size=config['batch_size'], pin_memory=True,
                                        shuffle=False)

test_loader = torch.utils.data.DataLoader(test_data, num_workers=2,
                                        batch_size=config['batch_size'], pin_memory=True,
                                        shuffle=False)


raw_eeg, LR_label, Angle_label, Amp_label, Pos_label, IsGenerated = next(iter(train_loader))

print("DATASET COMPLETED!!!!!!!")
input_shape = (129, 500)

output_LR = 1
output_Angle = 1
output_Amp = 1
output_Pos = 2

if config['architecture'] == 'Xception':
    model = Xception(input_shape, output_LR, output_Angle, output_Amp, output_Pos, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])

elif config['architecture'] == 'CNN':
    model = CNN(input_shape, output_LR, output_Angle, output_Amp, output_Pos, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])
        

elif config['architecture'] == 'PyramidalCNN':
    model = PyramidalCNN(input_shape, output_LR, output_Angle, output_Amp, output_Pos, kernel_size=40, nb_filters=64, depth=6, batch_size=config['batch_size'])


# print("input_shape", input_shape)

model = model.to(device)
summary(model,raw_eeg)

# criterion = nn.BCEWithLogitsLoss() if config['task']=='LR_task' else nn.MSELoss()
# if config['variable'] == 'Angle' and config['task']=='Direction_task':
#     criterion = angle_loss

# Losses
lr_criterion = nn.BCEWithLogitsLoss(reduce = False)
angle_criterion = angle_loss
amplitude_criterion = nn.MSELoss(reduce = False)
abs_pos_coriterion = nn.MSELoss(reduce = False)
criterion = [lr_criterion, angle_criterion, amplitude_criterion, abs_pos_coriterion]
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) #Defining Optimizer
# TODO: may change the scheduler later
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=0.0001, verbose=True)
# if config['task']=='LR_task':
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=0.0001, verbose=True)
scaler = torch.cuda.amp.GradScaler()

# TODO: may add wandb part later once after there is no bug in the code
torch.cuda.empty_cache()

epochs = config['epochs']
# TODO: change how to store the accuracies, store it in separate variables
best_acc = 0.0 # Monitor best accuracy in your run

# Initializing for early stopping 
best_val_meansure = 0.0
patience_count = 0 
patience_max = 20 # TODO: initialized based on paper


for epoch in range(config['epochs']):
    print("\nEpoch {}/{}".format(epoch+1, epochs))

    train_loss, lr_true_list, lr_pred_list, amp_true_list, amp_pred_list, pos_true_list, pos_pred_list, angle_true_list, angle_pred_list = train(model, optimizer, criterion, scaler, train_loader, weights)
    print("\tTrain Loss: {:.4f}".format(train_loss))
    print("\tTrain:")

    train_measure_LR, train_pred_LR = get_output(lr_pred_list, lr_true_list, 'LR_task','LR')
    train_measure_amp, train_pred_amp = get_output(amp_pred_list, amp_true_list, 'Direction_task','Amplitude')
    train_measure_angle, train_pred_angle = get_output(angle_pred_list, angle_true_list, 'Direction_task','Angle')
    train_measure_pos, train_pred_pos = get_output(pos_pred_list, pos_true_list, 'Position_task','Position')

    lr_true_list, lr_pred_list, amp_true_list, amp_pred_list, pos_true_list, pos_pred_list, angle_true_list, angle_pred_list = eval(model, val_loader)
    print("\tValidation:")
    val_measure_LR, val_pred_LR = get_output(lr_pred_list, lr_true_list, 'LR_task','LR')
    val_measure_amp, val_pred_amp = get_output(amp_pred_list, amp_true_list, 'Direction_task','Amplitude')
    val_measure_angle, val_pred_angle = get_output(angle_pred_list, angle_true_list, 'Direction_task','Angle')
    val_measure_pos, val_pred_pos = get_output(pos_pred_list, pos_true_list, 'Position_task','Position')
    
    ## Early Stopping condition
    # if abs(eval_measure_LR - best_val_meansure) > 0.1:
    #     best_val_meansure = val_measure
    # else: 
    #     patience_count += 1

    # if patience_count  >= patience_max:
    #     print("\nValid Accuracy didn't improve since last {} epochs.", patience_count)
    #     break 



    ### Log metrics at each epoch in your run - Optionally, you can log at each batch inside train/eval functions (explore wandb documentation/wandb recitation)
    # wandb.log({"train loss": train_loss, "validation accuracy": accuracy})

    ### Save checkpoint if accuracy is better than your current best
    
    torch.save({'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'acc': [val_measure_LR, val_measure_amp, val_measure_angle, val_measure_pos]}, 
    '../checkpoints/'+config['architecture']+'_'+'multitask'+'_checkpoint.pth')

    
    # scheduler.step(val_measure)
#     ## Save checkpoint in wandb
    #    wandb.save('checkpoint.pth')

#     Is your training time very high? Look into mixed precision training if your GPU (Tesla T4, V100, etc) can make use of it 
#     Refer - https://pytorch.org/docs/stable/notes/amp_examples.html

# ## Finish your wandb run
# run.finish()

lr_true_list, lr_pred_list, amp_true_list, amp_pred_list, pos_true_list, pos_pred_list, angle_true_list, angle_pred_list = test(model, test_loader)
print("\tTest:")
test_measure_LR, test_pred_LR = get_output(lr_pred_list, lr_true_list, 'LR_task','LR')
test_measure_amp, test_pred_amp = get_output(amp_pred_list, amp_true_list, 'Direction_task','Amplitude')
test_measure_angle, test_pred_angle = get_output(angle_pred_list, angle_true_list, 'Direction_task','Angle')
test_measure_pos, test_pred_pos = get_output(pos_pred_list, pos_true_list, 'Position_task','Position')

results_name = '../results/'+config['architecture']+'_'+'multitask'+".npz"
print(results_name)
np.savez(results_name, 
        pred_LR = test_pred_LR, 
        truth_LR = lr_true_list, 
        measure_LR = test_measure_LR,

        pred_amp = test_pred_amp, 
        truth_amp = amp_true_list, 
        measure_amp = test_measure_amp,

        pred_angle = test_pred_angle, 
        truth_angle = angle_true_list,
         measure_angle = test_measure_angle,

         pred_pos = test_pred_pos, 
         truth_pos = pos_true_list, 
         measure_pos = test_measure_pos )


