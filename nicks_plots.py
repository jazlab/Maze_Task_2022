## Imports

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import lines as mlp_lines
from matplotlib import patches as mpl_patches
import numpy as np
import os
import pandas as pd
import seaborn as sns

## Load data

nn_path = os.path.join(os.getcwd(), 'nearest_neighbors_grid_slice.pickle')
nn_data = np.array(np.load(nn_path, allow_pickle=True))
area_path = os.path.join(os.getcwd(), 'area_grid_slice.pickle')
area_data = np.array(np.load(area_path, allow_pickle=True))
eye_path = os.path.join(os.getcwd(), 'all_eye_movements.pickle')
eye_data = np.load(eye_path, allow_pickle=True)

## Get metric dataframe

nn_human = 3.8694648235166644
nn_baseline = 5.056780818472368
area_human = 180.74561953437896
area_baseline = 239.2094445226091

nn_human /= nn_baseline
area_human /= area_baseline

metric_df_dict = {
    'metric': [],
    'subject': [],
    'Model': [],
    'replica': [],
    'Normalized Score': [],
}
for metric, data in [('nearest_neighbors', nn_data), ('area_between_paths', area_data)]:
    for replica, replica_data in enumerate(data):
        for model, model_data in zip(['EXIT', 'SIM', 'HYBRID'], replica_data):
            for subject, value in enumerate(model_data):
                metric_df_dict['metric'].append(metric)
                metric_df_dict['subject'].append(subject)
                metric_df_dict['Model'].append(model)
                metric_df_dict['replica'].append(replica)
                scaling_factor = (
                    nn_baseline if metric == 'nearest_neighbors' else area_baseline
                )
                metric_df_dict['Normalized Score'].append(value / scaling_factor)

metric_df = pd.DataFrame(metric_df_dict)

palette = {'EXIT': 'r', 'SIM': 'b', 'HYBRID': 'g'}

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)




######################################################################
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=metric_df, x='Model', y='Normalized Score', hue='metric',
    palette=sns.color_palette(['#B876F1', '#5C30F2']),
    capsize=.1,
)
ax.hlines(y=1.0, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='--', color='k')
ax.hlines(y=nn_human, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='--', color='orange')
ax.hlines(y=area_human, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='--', color='orange')
ax.hlines(y=-999, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='dotted', color='orange')
ax.set_xlim([-0.5, 2.5])
# ax.legend(loc='upper center', title='factors')

# Set xticklabel colors
[t.set_color(i) for (i,t) in
 zip([(1, 0, 0), (0, 0, 1), (1, 0, 1)],ax.xaxis.get_ticklabels())]


# Create a custom legend
nn_facecolor = ax.legend_.get_patches()[0].get_facecolor()
area_facecolor = ax.legend_.get_patches()[1].get_facecolor()
start_point = [
    mpl_patches.Patch(facecolor=nn_facecolor, label='Nearest Neighbors'),
    mpl_patches.Patch(facecolor=area_facecolor, label='Area Between Paths'),
    mpl_patches.Patch(facecolor='w', label=''),
    mpl_patches.Patch(facecolor='w', label='    Baselines'),
    mlp_lines.Line2D([], [], color='k', lw=2, linestyle='--',
                     label='Baseline Model'),
    mlp_lines.Line2D([], [], color='orange', lw=2, linestyle='--',
                     label='Between Humans'),
    mlp_lines.Line2D([], [], color='orange', lw=2, linestyle='dotted',
                     label='Within Human'),
]
custom_lines = start_point
ax.legend(handles=custom_lines, bbox_to_anchor=(1.55, 1), loc='upper right', title='Metrics')
######################################################################





subjects = list(eye_data.keys())
print(subjects)

def _get_trial_saccade_vecs(x):
    if x.shape[0] > 1:
        s = x[1:] - x[:-1]
        s = s[np.linalg.norm(s, axis=1) < np.sqrt(2) * 39.]
        s = s * 14. / 39.
        if x.shape[0] < 1:
            return None
        else:
            return s
    else:
        return None

def _get_subject_saccade_vecs(v):
    trial_saccade_vecs = [_get_trial_saccade_vecs(np.array(x)) for x in v]
    trial_saccade_vecs = [x for x in trial_saccade_vecs if x is not None]
    subject_saccade_vecs = np.concatenate(trial_saccade_vecs)
    return subject_saccade_vecs

saccade_vectors = {k: _get_subject_saccade_vecs(v) for k, v in eye_data.items()}

saccade_df_dict = {
    'subject': [],
    'x': [],
    'y': [],
}
for k, v in saccade_vectors.items():
    num_saccades = v.shape[0]
    saccade_df_dict['subject'].extend(num_saccades * [k])
    saccade_df_dict['x'].extend(v[:, 0].tolist())
    saccade_df_dict['y'].extend(v[:, 1].tolist())

saccade_df = pd.DataFrame(saccade_df_dict)

all_saccades = {
    'Human': np.concatenate([
            v for k, v in saccade_vectors.items()
            if k not in ['EXIT0.2', 'SIM0.2', 'HYBRID0.2', 'REJSAMP']
        ]),
    'EXIT': saccade_vectors['EXIT0.2'],
    'SIM': saccade_vectors['SIM0.2'],
    'HYBRID': saccade_vectors['HYBRID0.2'],
    'Baseline': saccade_vectors['REJSAMP'],
}

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
title_colors = {
    'Human': 'orange',
    'Baseline': 'k',
    'EXIT': (1, 0, 0),
    'SIM': (0, 0, 1),
    'HYBRID': (1, 0, 1),
}

num_samples = 1000
fig, axes = plt.subplots(ncols=len(all_saccades), figsize=(20, 4))
fig.tight_layout()
for ax, (k, v) in zip(axes, all_saccades.items()):
    inds = np.random.choice(np.arange(v.shape[0]), (num_samples,), replace=False)
    sub_data = v[inds]
    ax.scatter(sub_data[:, 0], sub_data[:, 1], s=10)
    ax.set_title(k, color=title_colors[k], size=25)
    ax.set_xlim([-8.5, 8.5])
    ax.set_ylim([-8.5, 8.5])
    if k == 'Human':
        ax.set_xlabel('horizontal degrees', size=20)
        ax.set_ylabel('vertical degrees', size=20)
