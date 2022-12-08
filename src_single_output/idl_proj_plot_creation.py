import numpy as np
import matplotlib.pyplot as plt
import random
### -------------------------#############
# Loading the Data
raw_data=np.load('D:\Masters USA\Pitts\Fall_2022\IDL_11-785\Project_IDL\Git_EEG_saccade_detection\DeepLearningProject-11785\data\Position_task_with_dots_synchronised_min.npz')
EEG=raw_data['EEG']
label=raw_data['labels']
time=np.arange(0,1,0.002)
print(len(time))
print(raw_data['labels'][1000,:])
EEG_Sam=EEG[50,:,:]
label_Sam=label[50][1:]

########### -----------------EEG PLot--------------##############
# Setting Figure Size
plt.rcParams["figure.figsize"] = [4.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(20)
fig.suptitle('20 Channels, Occular EEG Montage')
Ch=np.random.choice(range(129),size=20, replace=False)
Ch_Clr=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(20):
    axs[i].plot(time, EEG_Sam[:,Ch[i]],Ch_Clr[i])
    axs[i].grid(True)
    # axs[i].grid(axis='y', color='0.95')
    # axs[i].axis('off')
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].get_yaxis().set_ticks([])
    if i<19: axs[i].get_xaxis().set_ticks([])
plt.show()

########### -----------------Display PLot--------------##############
print(label_Sam)
