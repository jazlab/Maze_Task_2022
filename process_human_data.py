import os
import json
import numpy as np
import scipy.stats
import torch

from settings import set_eval_parameters

MAZE_STIMULI_DIR, HUMAN_DATA_DIR, EXCLUDE_AMPLITUDE_UNDER, EXCLUDE_AFTER_TARGDIST_UNDER, REJSAMP_DATA_FILE = set_eval_parameters()

device = 'cpu'


def filter_saccade_amplitudes(fixations, exclude_amplitude_under, exclude_after_targdist_under, target):
    if len(fixations) <= 1:
        return fixations
    index = 0
    while index < len(fixations)-1:
        if np.linalg.norm(fixations[index+1]-fixations[index]) < exclude_amplitude_under:
            fixations = np.delete(fixations, index+1, axis=0)
        else:
            index += 1
    close_to_targ = np.where(np.linalg.norm(fixations-target, axis=1) < exclude_after_targdist_under)[0]
    fixations = fixations[:close_to_targ[0]+1]if close_to_targ.size > 0 else fixations
    return fixations


# fetch test stimuli
def get_test_stimuli(device):
    maze_stimuli = []
    maze_paths = []
    for maze_number in range(1, 400+1):
        with open('{}/{}'.format(MAZE_STIMULI_DIR, str(maze_number).rjust(5, '0')), 'r') as f:
            maze_data = json.load(f)
        maze = np.ones((39, 39))
        for wall in maze_data[3]:
            maze[max(39-2*wall[1][1], 0):(39-2*wall[0][1]+1), max((2*wall[0][0]-1), 0):(2*wall[1][0])] = 0
        maze_path_indices = 2 * np.array(maze_data[2])
        maze_path_indices[:, 1] = 38 - maze_path_indices[:, 1]
        maze_stimuli.append(maze)
        maze_paths.append(maze_path_indices)
    maze_stimuli = np.array(maze_stimuli)

    test_mazes = torch.as_tensor(maze_stimuli).float().to(device)
    test_exits = torch.as_tensor([path[-1] for path in maze_paths]).float().to(device)
    test_entrances = torch.as_tensor([path[0] for path in maze_paths]).float().to(device)
    test_paths = maze_paths
    return test_mazes, test_entrances, test_exits, test_paths


test_mazes, test_entrances, test_exits, test_paths = get_test_stimuli(device)
subjects = ['S{}'.format(i) for i in range(1, 14+1)]


def get_human_eye_poss(subject):
    """Fetch human eye movement data."""
    pair_maze_ids = [None] + [-1] * 400
    subject_all_eye_poss = []
    for maze_number in range(1, 400 + 1):
        if not os.path.exists('{}/{}/{}.json'.format(HUMAN_DATA_DIR, subject, maze_number)):
            subject_all_eye_poss.append(np.array([[1e9, 1e9]]))
            continue
        with open('{}/{}/{}.json'.format(HUMAN_DATA_DIR, subject, maze_number), 'r') as f:
            subject_data = json.load(f)

        if -1 in pair_maze_ids:
            pair_maze_ids[maze_number] = subject_data['pair_maze_id']

        eye_poss_indices = np.dstack((subject_data['eye_x'], 1-np.array(subject_data['eye_y'])))[0]

        # scale the data to the maze
        left, right = -0.5-(0.15/0.7*39), 38.5+(0.15/0.7*39)
        eye_poss_indices[:,0] = left + (right-left) * eye_poss_indices[:,0]
        top, bottom = -0.5-(0.2/0.7*39), 38.5+(0.1/0.7*39)
        eye_poss_indices[:,1] = top + (bottom-top) * eye_poss_indices[:,1]

        saccade_flags = subject_data['flag_saccade']
        fixation_indices = np.where(~np.array(saccade_flags))[0]

        if len(fixation_indices) < 2:
            subject_all_eye_poss.append(np.array([[1e9, 1e9]]))
            continue

        starts = np.concatenate(([0], np.where(np.ediff1d(fixation_indices) > 1)[0] + 1))
        ends = np.concatenate((np.where(np.ediff1d(fixation_indices) > 1)[0] + 1, [fixation_indices.size]))
        indices_by_fixation = [fixation_indices[s:e] for s, e in zip(starts, ends) if e-s >= 100]
        centers_of_mass = np.array([np.mean(eye_poss_indices[si], axis=0) for si in indices_by_fixation])
        centers_of_mass = filter_saccade_amplitudes(centers_of_mass, EXCLUDE_AMPLITUDE_UNDER, EXCLUDE_AFTER_TARGDIST_UNDER, test_exits[maze_number-1].numpy())
        subject_all_eye_poss.append(centers_of_mass if centers_of_mass.size > 0 else np.array([[1e9, 1e9]]))

    return subject_all_eye_poss, pair_maze_ids


# get saccade summary distributions for human data
def angle(p1, p2, p3):
    """Angle going from p2 to p1 to p3."""
    v1 = (p1[0]-p2[0], p1[1]-p2[1])
    v2 = (p3[0]-p1[0], p3[1]-p1[1])
    return np.arctan2(v1[0]*v2[1] - v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1])
def saccade_angle_distr(all_eye_poss):
    return [[np.degrees(angle(eye_poss[i], eye_poss[i-1], eye_poss[i+1])) for i in range(1, len(eye_poss)-1)] for j,eye_poss in enumerate(all_eye_poss)]
def saccade_amplitude_distr(all_eye_poss):
    return [[np.linalg.norm(eye_poss[i]-eye_poss[i+1]) for i in range(len(eye_poss)-1)] for eye_poss in all_eye_poss]
def flat_saccade_angle_distr(all_eye_poss):
    return np.array([ang for angs in saccade_angle_distr(all_eye_poss) for ang in angs])
def flat_saccade_amplitude_distr(all_eye_poss):
    return np.array([amp for amps in saccade_amplitude_distr(all_eye_poss) for amp in amps])

def get_human_saccade_kernels():
    human_eye_poss = np.concatenate([get_human_eye_poss(subj)[0] for subj in subjects])
    human_amplitudes = flat_saccade_amplitude_distr(human_eye_poss)
    human_angles = flat_saccade_angle_distr(human_eye_poss)
    human_num_saccades = np.array([len(poss) for poss in human_eye_poss])

    amplitude_kernel = scipy.stats.gaussian_kde(human_amplitudes)
    angle_kernel = scipy.stats.gaussian_kde(human_angles)
    num_saccades_kernel = scipy.stats.gaussian_kde(human_num_saccades)

    return amplitude_kernel, angle_kernel, num_saccades_kernel


def human_sim_speed():
    """Estimate human simulation speed from human eye data."""
    human_eye_poss = {}
    for subj in subjects:
        human_eye_poss[subj], _ = get_human_eye_poss(subj)
    maze_lengths = [len(p) * 2 for p in test_paths * len(subjects)]
    num_saccades = [len(p) - 1 for subj in subjects for p in human_eye_poss[subj]]
    sim_speeds = [l/s for l, s in zip(maze_lengths, num_saccades) if s > 0]
    return round(np.mean(sim_speeds))
