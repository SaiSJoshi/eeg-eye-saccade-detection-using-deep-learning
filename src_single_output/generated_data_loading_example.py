import numpy as np
data_LR = np.load('./data/Position_task_with_dots_synchronised_min.npz', allow_pickle = True)
print(data_LR.files)
print(data_LR['LR'].item()['IsGenerated'])
print(data_LR['LR'].item()['label'])