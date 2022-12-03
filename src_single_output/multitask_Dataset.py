import numpy as np
import torch
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

def save_label(data_path, variable, IsGenerated, generated_label = None):
    whole_data = np.load(data_path, allow_pickle=True)
    whole_data = dict(whole_data)

    if IsGenerated == False:
        whole_label = whole_data['labels']
        whole_label = whole_label[:, 1:]
        if variable == 'Angle':
            whole_label = whole_label[:,1]
        elif variable == 'Amplitude':
            whole_label = whole_label[:,0]
    else:
        whole_label = generated_label
        
    whole_data[variable] = {"IsGenerated": IsGenerated,
                            "label": whole_label
                            }
    
    np.savez(data_path,**whole_data)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, hilbert, train_ratio, val_ratio, test_ratio, task, variable, partition):
        
        max_ids = 0
        whole_eeg = []
        whole_LR_label = []
        Angle_label = []
        Amp_label = []
        Pos_label = []
        whole_IsGenerated = []
        whole_ids = []
        for i in len(data_path):
            whole_data = np.load(data_path[i], allow_pickle=True)
            eeg = whole_data['EEG']
            labels = whole_data['labels']
            if hilbert == False:
                eeg = eeg.transpose((0,2,1))
            
            whole_eeg.append(eeg)
            whole_LR_label.append(whole_data['LR'].item()['label'])
            Angle_label.append(whole_data['Angle'].item()['label'])
            Amp_label.append(['Amplitude'].item()['label'])
            Pos_label.append(whole_data['Position'].item()['label'])

            IsGenerated = np.array([whole_data['LR'].item()['IsGenerated'], whole_data['Angle'].item()['IsGenerated'],
            whole_data['Amplitude'].item()['IsGenerated'], whole_data['Position'].item()['IsGenerated']])
            whole_IsGenerated.append(IsGenerated)

            ids = labels[:, 0]+max_ids
            max_ids = max(ids)
            whole_ids.append(ids)

            if task == 'Direction_task':
                if variable == 'Angle':
                    whole_label = whole_label[:,1]
                elif variable == 'Amplitude':
                    whole_label = whole_label[:,0]
            elif task == 'Position_task':
                pass

        whole_eeg = np.concatenate(whole_eeg, axis=0)
        whole_LR_label = np.concatenate(whole_LR_label, axis=0)
        # label_min = np.min(whole_label)
        # label_max = np.max(whole_label)

        # whole_label = (whole_label - label_min)/(label_max - label_min)
        # self.label_min = label_min
        # self.label_max = label_max

        # Split the data
        train_idx, val_idx, test_idx = split(
            ids, train_ratio, val_ratio, test_ratio)

        if (partition == "train"):
            EEG = whole_eeg[train_idx]
            label = whole_label[train_idx]
        elif (partition == "val"):
            EEG = whole_eeg[val_idx]
            label = whole_label[val_idx]
        else:
            EEG = whole_eeg[test_idx]
            label = whole_label[test_idx]

        self.EEG = EEG  # (n,1,258)
        # (n,1) # Left_Right # Angle /Amplitude (n,2) # position (n,2)
        self.label = label

        # Making sure that we have the same no. of labels and trials
        assert len(self.EEG) == len(self.label)

        self.length = len(self.EEG)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        raw_eeg = self.EEG[ind]  # (1,258)

        raw_eeg = torch.FloatTensor(raw_eeg)  # Convert to tensors
        labels = torch.tensor(self.label[ind])

        return raw_eeg, labels
        # raw_egg [500*128]
        # lr, angle, amplitue, position(1,2)
        # IsGenerated [1,4]

class TestDataset(torch.utils.data.Dataset): # for generating labels

    def __init__(self, data_path):

        whole_data = np.load(data_path, allow_pickle=True)
        whole_eeg = whole_data['EEG']
        EEG = whole_eeg.transpose((0,2,1))

        self.EEG = EEG  # (n,1,258)
        # (n,1) # Left_Right # Angle /Amplitude (n,2) # position (n,2)

        self.length = len(self.EEG)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        signal = self.EEG[ind]  # (1,258)

        return torch.FloatTensor(signal)
