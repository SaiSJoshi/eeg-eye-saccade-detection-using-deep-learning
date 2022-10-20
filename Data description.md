# Left-right
## Files
- "LR_task_with_antisaccade_synchronised_max.npz"

'EEG' (30825,500,129) (# of trials, # of time samples, # of channels)<br /> 'labels' (30825,2) (first column refers to IDs, second column refers to labels)

- "LR_task_with_antisaccade_synchronised_min.npz"

'EEG' (30842,500,129) (# of trials, # of time samples, # of channels)<br /> 'labels' (30842,2) (first column refers to IDs, second column refers to labels)

- "LR_task_with_antisaccade_synchronised_max_hilbert.npz"

'EEG' (30825,1,258) (# of trials, each trials contains phase and amplitude information)<br /> 'labels' (30825,2) (first column refers to IDs, second column refers to labels)

- "LR_task_with_antisaccade_synchronised_min_hilbert.npz"

'EEG' (30842,1,258) (# of trials, each trials contains phase and amplitude information)<br /> 'labels' (30842,2) (first column refers to IDs, second column refers to labels)



## Input/Output
train:validation:test = 0.7:0.15:0.15 (split based on IDs, same ID goes to the same group) <br />
Input: minimally reprocessed hilbert data; Output: Left/Right <br />
performance metrics: accuracy_score <br />

# Position task with dots

## Files
- "Position_task_with_dots_synchronised_min.npz"

'EEG' (21464, 500, 129) (# of trials, # of time samples, # of channels)<br />
'labels' (21464, 3) (first column refers to IDs, second column refers to labels)<br />

- "Position_task_with_dots_synchronised_min_hilbert.npz"

'EEG' (21464, 1, 258) (# of trials, # of time samples, # of channels)<br />
'labels' (21464, 3) (first column refers to IDs, second column refers to labels)<br />


- "Position_task_with_dots_synchronised_max.npz"

'EEG' (21659, 500, 129) (# of trials, # of time samples, # of channels)<br />
'labels' (21464, 3) (first column refers to IDs, second column refers to labels)<br />

- "Position_task_with_dots_synchronised_max_hilbert.npz"

'EEG' (21659, 1, 258) (# of trials, # of time samples, # of channels) <br />
'labels' (21659, 3) (first column refers to IDs, second column refers to labels) <br />

## Input/Output
train: validation:test = <br />
Input: minimally reprocessed hilbert data <br />
Output: abs position <br />
Model Type : Regression <br />


