import numpy as np
import torch
import math


def split(ids, train, val, test):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
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

        whole_data = np.load(data_path, allow_pickle=True)
        whole_eeg = whole_data['EEG']
        whole_label = whole_data['labels']
        ids = whole_label[:, 0]
        whole_label = whole_label[:, 1:]
        if hilbert == False:
            whole_eeg = whole_eeg.transpose((0,2,1))

        if task == 'Direction_task':
            if variable == 'Angle':
                whole_label = whole_label[:,1]
            elif variable == 'Amplitude':
                whole_label = whole_label[:,0]
        elif task == 'Position_task':
            pass

        
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

        frames = self.EEG[ind]  # (1,258)

        frames = torch.FloatTensor(frames)  # Convert to tensors
        phoneme = torch.tensor(self.label[ind])

        return frames, phoneme

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
