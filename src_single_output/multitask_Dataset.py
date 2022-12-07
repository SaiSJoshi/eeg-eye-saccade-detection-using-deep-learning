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
    def __init__(self, data_path):
        whole_data = np.load(data_path, allow_pickle=True)
        self.EEG = whole_data["EEG"]
        self.LR_label = whole_data["LR_label"]
        self.Angle_label = whole_data["Angle_label"]
        self.Amp_label = whole_data["Amp_label"]
        self.Pos_label = whole_data["Pos_label"]
        self.IsGenerated = whole_data["IsGenerated"]

        range_LR = [np.min(self.LR_label[np.where(self.IsGenerated[:,0] == False)]), np.max(self.LR_label[np.where(self.IsGenerated[:,0] == False)])]
        range_Angle = [np.min(self.Angle_label[np.where(self.IsGenerated[:,1] == False)]), np.max(self.Angle_label[np.where(self.IsGenerated[:,1] == False)])]
        range_Amp = [np.min(self.Amp_label[np.where(self.IsGenerated[:,2] == False)]), np.max(self.Amp_label[np.where(self.IsGenerated[:,2] == False)])]
        range_Pos = [np.min(self.Pos_label[np.where(self.IsGenerated[:,3] == False)]), np.max(self.Pos_label[np.where(self.IsGenerated[:,3] == False)])]
        self.scaling = np.stack([range_LR, range_Angle, range_Amp, range_Pos], axis = 0)
    
        assert len(self.EEG) == len(self.IsGenerated)

        self.length = len(self.EEG)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        raw_eeg = self.EEG[ind]  # (1,258)

        raw_eeg = torch.FloatTensor(raw_eeg)  # Convert to tensors
        LR_label = torch.tensor(self.LR_label[ind])
        Angle_label = torch.tensor(self.Angle_label[ind])
        Amp_label = torch.tensor(self.Amp_label[ind])
        Pos_label = torch.tensor(self.Pos_label[ind])
        IsGenerated = torch.tensor(self.IsGenerated[ind])

        return raw_eeg, LR_label, Angle_label, Amp_label, Pos_label, IsGenerated


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
