import numpy as np
import mne  # If this line returns an error, uncomment the following line
# !easy_install mne --upgrade
import matplotlib.pyplot as plt
# import plotly.plotly as py
from chart_studio import plotly as py
from plotly import tools
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
from plotly.offline import iplot
import cufflinks as cf

# Loading the Data
raw_data=np.load('D:\Masters USA\Pitts\Fall_2022\IDL_11-785\Project_IDL\Git_EEG_saccade_detection\DeepLearningProject-11785\data\Position_task_with_dots_synchronised_min.npz')
EEG1=raw_data['EEG']
label=raw_data['labels']
time=np.arange(0,1,0.002)
print(len(time))
print(raw_data['labels'][1000,:])
EEG=EEG1[1,:,:]
# add plot inline in the page
# matplotlib inline

mne.set_log_level('WARNING')

# Creating the EEG plot
n_channels = 20
step = 1. / n_channels
kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

# create objects for layout and traces
layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
traces = [Scatter(x=time, y=EEG.T[:, 0])]

# loop over the channels
for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
        traces.append(Scatter(x=time, y=EEG.T[:, ii], yaxis='y%d' % (ii + 1)))

ch_names = ["EEG1","EEG2","EEG3","EEG4","EEG5","EEG6","EEG7","EEG8","EEG9","EEG10",
            "EEG11","EEG12","EEG13","EEG14","EEG15","EEG16","EEG17","EEG18","EEG19","EEG20",]

# add channel names using Annotations
annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                      text=ch_name, font=Font(size=9), showarrow=False)
                          for ii, ch_name in enumerate(ch_names)])
layout.update(annotations=annotations)

# set the size of the figure and plot it
layout.update(autosize=False, width=1000, height=600)
fig = Figure(data=(traces), layout=layout)
cf.go_offline() #will make cufflinks offline
cf.set_config_file(offline=False, world_readable=True)
py.iplot(fig, filename='shared xaxis')