import numpy as np
import math

def split(ids, train, val, test):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    IDs = np.random.permutation(IDs)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split+val_split:])

    return train, val, test

data_path = ['./data/LR_task_with_antisaccade_synchronised_min.npz',
    './data/Direction_task_with_dots_synchronised_min.npz',
    './data/Position_task_with_dots_synchronised_min.npz'
    ]

length = np.zeros(4, dtype = int)
for i in range(len(data_path)):
    whole_data = np.load(data_path[i], allow_pickle=True)
    eeg = whole_data['EEG']
    l, dim2, dim1 = eeg.shape
    length[i+1] = length[i]+l
    del whole_data
    del eeg

max_ids = 0
whole_eeg = np.zeros((int(length[-1]),dim1,dim2), dtype=np.float16)
whole_LR_label = []
whole_Angle_label = []
whole_Amp_label = []
whole_Pos_label = []
whole_IsGenerated = []
whole_ids = []

for i in range(len(data_path)):
    whole_data = np.load(data_path[i], allow_pickle=True)
    eeg = whole_data['EEG']
    labels = whole_data['labels']
    
    eeg = eeg.transpose((0,2,1))
    whole_eeg[length[i]:length[i+1],:,:] = eeg
    
    whole_LR_label.append(np.squeeze(whole_data['LR'].item()['label']))
    whole_Angle_label.append(np.squeeze(whole_data['Angle'].item()['label']))
    whole_Amp_label.append(np.squeeze(whole_data['Amplitude'].item()['label']))
    whole_Pos_label.append(np.squeeze(whole_data['Position'].item()['label']))

    IsGenerated = np.array([whole_data['LR'].item()['IsGenerated'], whole_data['Angle'].item()['IsGenerated'],
    whole_data['Amplitude'].item()['IsGenerated'], whole_data['Position'].item()['IsGenerated']])
    IsGenerated = np.broadcast_to(IsGenerated, (len(eeg), 4))
    print(IsGenerated.shape)
    whole_IsGenerated.append(IsGenerated)

    ids = labels[:, 0]+max_ids
    max_ids = max(ids)
    whole_ids.append(ids)

    del eeg
    del whole_data
    del labels
    print("pass")


# whole_eeg = np.concatenate(whole_eeg, axis=0)
whole_LR_label = np.concatenate(whole_LR_label, axis=0)
whole_Angle_label = np.concatenate(whole_Angle_label, axis=0)
whole_Amp_label = np.concatenate(whole_Amp_label, axis=0)
whole_Pos_label = np.concatenate(whole_Pos_label, axis=0)
whole_IsGenerated = np.concatenate(whole_IsGenerated, axis=0)
whole_ids = np.concatenate(whole_ids, axis=0)

train_idx, val_idx, test_idx = split( whole_ids, train=0.7, val=0.15, test=0.15)

np.savez("./data/Generated_train.npz", EEG = whole_eeg[train_idx], LR_label = whole_LR_label[train_idx], Angle_label = whole_Angle_label[train_idx],
            Amp_label = whole_Amp_label[train_idx], Pos_label = whole_Pos_label[train_idx], 
            IsGenerated = whole_IsGenerated[train_idx])

np.savez("./data/Generated_val.npz", EEG = whole_eeg[val_idx], LR_label = whole_LR_label[val_idx], Angle_label = whole_Angle_label[val_idx],
            Amp_label = whole_Amp_label[val_idx], Pos_label = whole_Pos_label[val_idx], 
            IsGenerated = whole_IsGenerated[val_idx])

np.savez("./data/Generated_test.npz", EEG = whole_eeg[test_idx], LR_label = whole_LR_label[test_idx], Angle_label = whole_Angle_label[test_idx],
            Amp_label = whole_Amp_label[test_idx], Pos_label = whole_Pos_label[test_idx], 
            IsGenerated = whole_IsGenerated[test_idx])