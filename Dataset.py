
data=np.load('data\LR_Task.npz')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, partition): # Feel free to add more arguments
        
        if (partition=="train"):
            data=np.random(data,70) # change this 
        elif (partition=="val"):
            data=np.random(data,70)
        else:
            data=np.random(data,70)


        self.EEG = data['EEG']  # (n,1,258)  
        self.label=data['label'] # (n,2) # Left_Right # Angle /Amplitude (n,3) # position (n,3)
        
        assert len(self.EEG) == len(self.label) # Making sure that we have the same no. of mfcc and transcripts    
       
        self.length = len(self.EEG)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        
        frames = self.EEG[ind] #(1,258)
        
        frames = torch.FloatTensor(frames) # Convert to tensors
        phoneme = torch.tensor(self.label[ind,1:])       

        return frames, phoneme


# add bench mark here


train_data = Dataset(data,partition="train",)
val_data = Dataset(data,partition="val",)
test_data = Dataset(data,partition="test",)



train_loader = torch.utils.data.DataLoader(train_data, num_workers= 4,
                                           batch_size=config['batch_size'], pin_memory= True,
                                           shuffle= True)

val_loader = torch.utils.data.DataLoader(val_data, num_workers= 2,
                                         batch_size=config['batch_size'], pin_memory= True,
                                         shuffle= False)

test_loader = torch.utils.data.DataLoader(test_data, num_workers= 2, 
                                          batch_size=config['batch_size'], pin_memory= True, 
                                          shuffle= False)