import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import lines as mlp_lines
from matplotlib import patches as mpl_patches
import seaborn as sns
import similaritymeasures

from settings import set_eval_parameters, set_task_parameters, set_gazeRNN_hyperparameters
from gazeRNN import SaccadeCNN, MemCNN, run_batch
from process_human_data import filter_saccade_amplitudes, get_human_eye_poss, get_test_stimuli

device = 'cpu'

MAZE_STIMULI_DIR, HUMAN_DATA_DIR, EXCLUDE_AMPLITUDE_UNDER, EXCLUDE_AFTER_TARGDIST_UNDER, REJSAMP_DATA_FILE = set_eval_parameters()

all_eye_poss = {}

test_mazes, test_entrances, test_exits, test_paths = get_test_stimuli(device)

# fetch human data
subjects = ['S{}'.format(i) for i in range(1, 14+1)]
pair_maze_ids = [-1] * (len(test_mazes) + 1)
for subj in subjects:
    all_eye_poss[subj], pair_maze_ids_subj = get_human_eye_poss(subj)
    for i,_ in enumerate(pair_maze_ids):
        if pair_maze_ids[i] == -1 and pair_maze_ids_subj[i] != -1:
            pair_maze_ids[i] = pair_maze_ids_subj[i]

# run models on test stimuli
test_models = np.array([
    ['EXIT0.2', 'exit0.2', 1850000],
    ['SIM0.2', 'sim0.2', 1850000],
    ['HYBRID0.2', 'hybrid0.2', 1850000],
    ])
models = test_models[:, 0].tolist()


def load_and_run_model(version, iteration):
    MAZE_SIZE, MAZE_DIR, NUM_LAYERS, PIXELS_PER_SQUARE, IMG_DIM = set_task_parameters()
    SPATIAL_BASIS, TAU, NOISE_STDEV, N_SACCADES, EXIT_BETA, SIM_BETA, SIM_SPEED, MEM_CHANNELS, BATCH_SIZE, LEARNING_RATE = set_gazeRNN_hyperparameters()
    if version == 'exit0.2':
        EXIT_BETA, SIM_BETA = 1, 0
    elif version == 'sim0.2':
        EXIT_BETA, SIM_BETA = 0, 1
    else:
        EXIT_BETA, SIM_BETA = 1, 2

    saccadecnn = SaccadeCNN(MEM_CHANNELS, do_sim=(SIM_BETA > 0)).to(device)
    memcnn = MemCNN(MEM_CHANNELS).to(device)
    saccadecnn.load_state_dict(torch.load(os.path.join('saved_models', 'saccadecnn-{}-it{}.pt'.format(version, iteration)), map_location=torch.device('cpu')))
    memcnn.load_state_dict(torch.load(os.path.join('saved_models', 'memcnn-{}-it{}.pt'.format(version, iteration)), map_location=torch.device('cpu')))

    test_loss, test_eye_poss, test_ball_poss, test_true_ball_poss = run_batch('test', saccadecnn, memcnn, test_mazes, test_entrances, test_exits, test_paths)
    test_all_eye_poss = np.concatenate((test_entrances.cpu().reshape(-1,1,2), test_eye_poss), axis=1)
    test_all_ball_poss = np.concatenate((test_entrances.cpu().reshape(-1,1,2), test_ball_poss), axis=1)

    test_all_eye_poss_list = []
    for i, eye_poss in enumerate(test_all_eye_poss):
        test_all_eye_poss_list.append(filter_saccade_amplitudes(eye_poss, EXCLUDE_AMPLITUDE_UNDER, EXCLUDE_AFTER_TARGDIST_UNDER, test_exits[i].numpy()))

    return test_all_eye_poss_list, test_all_ball_poss


all_ball_poss = {}
for name, version, iter in test_models:
    test_all_eye_poss_list, test_all_ball_poss_list = load_and_run_model(version, iter)
    all_eye_poss[name] = test_all_eye_poss_list
    all_ball_poss[name] = test_all_ball_poss_list

with open(REJSAMP_DATA_FILE, 'rb') as f:
    rejsamp_eye_poss = pickle.load(f)
    all_eye_poss['REJSAMP'] = rejsamp_eye_poss

agents = list(all_eye_poss.keys())


def ax_plot_trial_sbs(ax, agent, maze_idx, all_eye_poss, all_ball_poss, plot_ball, force_square):
    test_maze = test_mazes[maze_idx].numpy()
    test_path = test_paths[maze_idx]
    test_targ = test_exits[maze_idx].numpy()
    test_all_eye_pos = all_eye_poss[agent][maze_idx]

    easier_on_eyes = test_maze.copy()
    easier_on_eyes[easier_on_eyes == 1] = 0.3
    easier_on_eyes[easier_on_eyes == 0] = 0.01

    ax.imshow(easier_on_eyes, cmap='gray_r', vmin=0, vmax=1)
    ax.plot(test_path[:, 0], test_path[:, 1], c='b', alpha=0.5)
    ax.plot(test_targ[0], test_targ[1], c='limegreen', marker='o', ms=10)
    if plot_ball:
        ax.plot(*all_ball_poss[agent][maze_idx].T, marker='o', c='y', mec='k', lw=3.5)
    ax.plot(*test_all_eye_pos.T, marker='o', c=agent_colors[agents_to_plot.index(agent)], mec='k', lw=2, ms=5)
    if force_square:
        ax.set_ylim(38.5, -0.5)
        ax.set_xlim(-0.5, 38.5)


agents_to_plot = ['S1', 'S2', 'EXIT0.2', 'SIM0.2', 'HYBRID0.2', 'REJSAMP']
agent_titles = ['Human 1', 'Human 2', 'EXIT', 'SIM', 'HYBRID', 'Baseline']
mazes_to_plot = [1, 6, 9]
agent_colors=['orange', 'orange', 'firebrick', 'b', 'magenta', 'k']
fig, axs = plt.subplots(len(agents_to_plot), len(mazes_to_plot), figsize=(6, 9))
for row in range(len(agents_to_plot)):
    for col in range(len(mazes_to_plot)):
        ax_plot_trial_sbs(axs[row, col], agents_to_plot[row], mazes_to_plot[col],
                          all_eye_poss, all_ball_poss, plot_ball=False, force_square=True)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
    axs[row, 0].set_ylabel(agent_titles[row], size=15, weight='demibold', labelpad=60, rotation=0)
fig.subplots_adjust(left=0.2, right=1, wspace=1.5, hspace=0.5)
fig.tight_layout()


def valid_eye_paths(eye_paths):
    """Returns true if all eye paths in the iterable are valid."""
    return np.all([1e9 not in eye_path for eye_path in eye_paths])


def nearest_distance(arr, pt):
    """Returns distance of the element in the array with smallest distance to the point."""
    return np.min([np.linalg.norm(pt - a) for a in arr])


def nn_path_dist(eye_poss1, eye_poss2):
    return np.mean([nearest_distance(eye_poss1, eye_pos2) for eye_pos2 in eye_poss2]), np.mean([nearest_distance(eye_poss2, eye_pos1) for eye_pos1 in eye_poss1])


def area_between(arr1, arr2):
    if len(arr1) == 1:
        arr1 = np.stack((arr1[0], arr1[0]))
    if len(arr2) == 1:
        arr2 = np.stack((arr2[0], arr2[0]))
    return similaritymeasures.area_between_two_curves(arr1, arr2)


def mean_path_dist(path_dist_func, agent1, agent2):
    path_dists = []
    processed = [False for _ in range(len(test_mazes))]
    for i in range(len(test_mazes)):
        if processed[i]:
            continue
        pair_i = pair_maze_ids[i + 1] - 1

        a1, a2 = all_eye_poss[agent1][i], all_eye_poss[agent2][i]
        a1p, a2p = all_eye_poss[agent1][pair_i], all_eye_poss[agent2][pair_i]

        # filter out cases with bad saccades (1e9 flag)
        if not valid_eye_paths((a1, a2, a1p, a2p)):
            continue
        if agent1 == agent2:
            path_dists.append(nn_path_dist(a1, a2p))
        else:
            path_dists.append(np.mean(np.array([path_dist_func(a1, a2),
                                                path_dist_func(a1, a2p),
                                                path_dist_func(a1p, a2),
                                                path_dist_func(a1p, a2p)]), axis=0))
        processed[i], processed[pair_i] = True, True
    return np.mean(path_dists)


def plot_mean_path_dist_grid(agents, names, path_dist_func):
    mean_path_dist_grid = [[mean_path_dist(path_dist_func, agents[a1], agents[a2]) for a2 in range(a1+1)]+[np.nan for _ in range(len(agents)-a1-1)] for a1 in range(len(agents))]
    distr_of = 'Nearest Neighbors Distances' if path_dist_func == nn_path_dist else 'Area Between Paths'
    plt.figure()
    plt.title('Mean {}'.format(distr_of), size=16)
    plt.xticks(range(len(names)), names, size=12, rotation=30)
    plt.yticks(range(len(names)), names, size=12)
    plt.imshow(mean_path_dist_grid, cmap='Reds')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    for i, row in enumerate(mean_path_dist_grid):
        for j, dist in enumerate(row):
            if not np.isnan(dist):
                plt.text(j, i, np.round(dist, 1) if path_dist_func == nn_path_dist else int(np.round(dist, 0)), size=8, va='center', ha='center')
    plt.show()


# plot_mean_path_dist_grid(agents, agents, nn_path_dist)
# plot_mean_path_dist_grid(agents, agents, area_between)


def repeat_metric(test_models, path_dist_func, outfile, N=100):
    """Evaluate models N times on a single metric."""
    mean_path_dist_grid_slices = []
    start_time = time.time()
    for rep in range(N):
        print('Rep {}, {:.2f}s'.format(rep, time.time()-start_time))
        for name, suff, iter in test_models:
            test_all_eye_poss_list, test_all_ball_poss_list = load_and_run_model(suff, iter)
            all_eye_poss[name] = test_all_eye_poss_list
        mean_path_dist_grid_slice = [[mean_path_dist(path_dist_func, agents[a1], agents[a2]) for a2 in range(0, len(subjects))] for a1 in range(len(subjects), len(agents)-1)]
        mean_path_dist_grid_slices.append(mean_path_dist_grid_slice)
    with open(outfile, 'wb') as f:
        pickle.dump(mean_path_dist_grid_slices, f)


# repeat_metric(test_models, nn_path_dist, 'nn_path_dist_100.pickle', N=100)
# repeat_metric(test_models, nn_path_dist, 'area_between_100.pickle', N=100)


##########################################################################
nn_data = np.array(np.load('nn_path_dist_100.pickle', allow_pickle=True))
area_data = np.array(np.load('area_between_100.pickle', allow_pickle=True))

# get metric dataframe
nn_baseline = 5.056780818472368
nn_human = 3.8694648235166644 / nn_baseline
area_baseline = 239.2094445226091
area_human = 180.74561953437896 / area_baseline

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
                scaling_factor = (nn_baseline if metric == 'nearest_neighbors' else area_baseline)
                metric_df_dict['Normalized Score'].append(value / scaling_factor)
metric_df = pd.DataFrame(metric_df_dict)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=metric_df, x='Model', y='Normalized Score', hue='metric',
    palette=sns.color_palette(['#B876F1', '#5C30F2']),
    capsize=.1,
)
ax.hlines(y=1.0, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='--', color='k')
ax.hlines(y=nn_human, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='--', color='orange')
ax.hlines(y=area_human, xmin=-0.5, xmax=2.5, linewidth=2, linestyle='--', color='orange')
ax.set_xlim([-0.5, 2.5])

# set xticklabel colors
[t.set_color(i) for (i,t) in zip([(1, 0, 0), (0, 0, 1), (1, 0, 1)], ax.xaxis.get_ticklabels())]

# create a custom legend
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

def get_trial_saccade_vecs(x):
    s = x[1:] - x[:-1]
    s = s[np.linalg.norm(s, axis=1) < np.sqrt(2) * 39.]
    s = s * 14. / 39.
    return s

saccade_vectors = {}
for k, v in all_eye_poss.items():
    saccade_vectors[k] = np.concatenate([get_trial_saccade_vecs(np.array(x)) for x in v if x.shape[0] > 1])

all_saccades = {
    'Human': np.concatenate([v for k, v in saccade_vectors.items() if k not in ['EXIT0.2', 'SIM0.2', 'HYBRID0.2', 'REJSAMP']]),
    'EXIT': saccade_vectors['EXIT0.2'],
    'SIM': saccade_vectors['SIM0.2'],
    'HYBRID': saccade_vectors['HYBRID0.2'],
    'Baseline': saccade_vectors['REJSAMP'],
}

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


