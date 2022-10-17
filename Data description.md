# Left-right
## Files
- "LR_task_with_antisaccade_synchronised_max.npz"

'EEG' (30825,500,129) (# of trials, # of time samples, # of channels); 'labels' (30825,2) (first column refers to IDs, second column refers to labels)

- "LR_task_with_antisaccade_synchronised_min.npz"

'EEG' (30842,500,129) (# of trials, # of time samples, # of channels); 'labels' (30842,2) (first column refers to IDs, second column refers to labels)

- "LR_task_with_antisaccade_synchronised_max_hilbert.npz"

'EEG' (30825,1,258) (# of trials, each trials contains phase and amplitude information); 'labels' (30825,2) (first column refers to IDs, second column refers to labels)

- "LR_task_with_antisaccade_synchronised_min_hilbert.npz"

'EEG' (30842,1,258) (# of trials, each trials contains phase and amplitude information); 'labels' (30842,2) (first column refers to IDs, second column refers to labels)

## Data processing
- 'min': the detection and interpolation of bad electrodes, and filtering the data with 40 Hz high- pass filter and 0.5 Hz low-pass filter; remove artifacts using ICA (ocular artifacts)
- 'max': same process as min processing but removes a much larger number of artifacts (muscles, heart, eyes, line noise, channel noise)
- 'hilbert': applied after band-passing the signal, resulting in a complex time series from which we extract phase and amplitude

## Input/Output
train:validation:test = 0.7:0.15:0.15 (split based on IDs, same ID goes to the same group)
Input: minimally reprocessed hilbert data; Output: Left/Right
performance metrics: accuracy_score
