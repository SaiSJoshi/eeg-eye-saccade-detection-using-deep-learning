# Data Annotation
- **Saccades** are rapid, ballistic eye movements that instantly change the gaze position. 
- **Fixations** are defined as time periods without saccades
- **Blinks** are considered a special case of fixation, where the pupil diameter is zero. 

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
Performance Metrics: accuracy_score <br />


# Angle/Amplitude
## Files
- "Direction_task_with_dots_synchronised_min.npz"

'EEG' (17982, 500, 129) (# of trials, # of time samples, # of channels)<br />
'labels' (17982, 3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_dots_synchronised_max_hilbert.npz"

'EEG' (17982, 1, 258) (# of trials, each trials contains phase and amplitude information)<br />
'labels' (17982, 3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_dots_synchronised_min.npz"

'EEG' (17830, 500, 129) (# of trials, # of time samples, # of channels)<br />
'labels' (17830,3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_dots_synchronised_min_hilbert.npz"

'EEG' (17830, 1, 258) (# of trials, each trials contains phase and amplitude information)<br />
'labels' (17830, 3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_processing_speed_synchronised_max.npz"

'EEG' (31191,500, 129) (# of trials, # of time samples, # of channels)<br />
'labels' (31191, 3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_processing_speed_synchronised_max_hilbert.npz"

'EEG' (31191, 1, 258) (# of trials, each trials contains phase and amplitude information)<br />
'labels' (31191, 3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_processing_speed_synchronised_min.npz"

'EEG' (31563, 500, 129) (# of trials, # of time samples, # of channels)<br />
'labels' (31563, 3) (first column refers to IDs, second column refers to labels)<br />

-	"Direction_task_with_processing_speed_synchronised_min_hilbert.npz"

'EEG' (31563, 1, 258) (# of trials, each trials contains phase and amplitude information)<br />
'labels' (31563, 3) (first column refers to IDs, second column refers to labels)<br />

## Input/Output
train:validation:test = 0.7:0.15:0.15 (split based on IDs, same ID goes to the same group)<br />
Input: minimally reprocessed hilbert data; Output: Angle and Amplitude<br />
Performance Metrics: RMSE<br />
- The naive baseline is given by the mean angle and amplitude in the training set and <br />amounts to 1.90 RMSE radians for the angle and 74.7 RMSE mm for the amplitude



# Absolute Position

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
train: validation:test = 0.7:0.15:0.15 (split based on IDs, same ID goes to the same group)  <br />
Input: minimally reprocessed hilbert data <br />
Output: abs position <br />
Task Type : Regression <br />


