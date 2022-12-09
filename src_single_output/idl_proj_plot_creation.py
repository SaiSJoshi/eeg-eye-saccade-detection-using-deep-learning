import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import time
import cv2
from PIL import Image
from numpy import asarray

### -----------Converting the frames of the image--------------#############
frames = []
video = cv2.VideoCapture("\Media1.mp4")
while True:
    read, frame= video.read()
    if not read:
        break
    frames.append(frame)
frames = np.array(frames)


### -------------------------#############
# Loading the Data
raw_data=np.load('Generated_test.npz')
EEG=raw_data['EEG']
label=np.load('CNN_2_8_unscaledMSEmultitask.npz')
n=10
test_pred_pos=label['pred_pos']
pos_true_list=label['truth_pos']


### -------------Figure Settings------------#############
k=0
grid=plt.GridSpec(20,2, wspace=0, hspace=0)
plt.figure(facecolor=(.18, .31, .31))
plt.suptitle('Deep Learning based EEG Saccade Detection',fontweight='bold',color='y')
ax=plt.subplot(grid[0:10,1])
ax3=plt.subplot(grid[10:,1])
while(k<=50):
    
    t=np.arange(0,1,0.002)
    id=np.random.randint(EEG.shape[0])
    EEG_Sam=EEG[id,:,:]
    label_true=pos_true_list[id,:]
    label_pred=test_pred_pos[id,:]

    ########### -----------------EEG PLot--------------##############

    # fig.suptitle('20 Channels, Occular EEG Montage')
    Ch=np.random.choice(range(129),size=20, replace=False)
    Ch_Clr=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(20):
        ax1=plt.subplot(grid[i,0])
        ax1.clear()
        ax1=plt.subplot(grid[i,0])
        ax1.set_facecolor('#eafff5')
        ax1.tick_params(labelcolor='peachpuff')
        plt.plot(t, EEG_Sam[Ch[i],:],Ch_Clr[i])
        plt.yticks([])
        if i==0:
            plt.title('Occular EEG Montage',fontweight='bold',color='y')
        plt.grid(visible=True, which='major', color='g', linestyle='--')
        if i==19:
            plt.xlabel('EEG Sample (1 Sec)',fontweight='bold',color='tab:orange')
        if i==9:
            plt.ylabel('EEG Amplitude (UV)',fontweight='bold',color='tab:orange')

        
    ######## ------------ Screen View--------############
    ax.clear()
    ax=plt.subplot(grid[0:10,1])
    ax.set_facecolor('#eafff5')


    plt.title('Detected Target',fontweight='bold',color='y')
    # True Label
    plt.scatter(label_true[0], label_true[1], marker="P",s=80,alpha=0.6,c='black')
    plt.text(label_true[0], label_true[1]+50, 'True', fontsize=10.0, fontweight='bold')
    # Predicted Label
    plt.scatter(label_pred[0], label_pred[1], marker="P",s=80,alpha=0.6,c='blue')
    plt.text(label_pred[0], label_pred[1]+50, 'Pred', fontsize=10.0, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    ax.tick_params(labelcolor='tab:orange')
    plt.xlim([-721,841])
    plt.ylim([-123,896])
        
    ######## ------------ Saccade View--------############
    ax3.clear()
    ax3=plt.subplot(grid[10:,1])
    plt.title('Saccade Movement',y=-0.01,fontweight='bold',color='0.7')
    img=plt.imread('D:\Masters USA\Pitts\Fall_2022\IDL_11-785\Project_IDL\saccade.png')
    # rotated_img = ndimage.rotate(img, 5*60)
    plt.imshow(frames[k*20][510:-80,500:1400,:],aspect='auto')
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.pause(0.5)
    k=k+1

    

