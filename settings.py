import numpy as np
import torch

MAZE_SIZE = 20
MAZE_DIR = 'layered_maze_lib/path_datasets/maze_size_{}/samples_per_pair_100_v0'.format(MAZE_SIZE)
NUM_LAYERS = 50  # large enough to cover most of the arena
PIXELS_PER_SQUARE = 2  # should be even
IMG_DIM = PIXELS_PER_SQUARE * MAZE_SIZE - 1
SPATIAL_BASIS = torch.as_tensor(np.meshgrid(np.arange(IMG_DIM), np.arange(IMG_DIM))).permute(1, 2, 0).to('cpu')

TAU = 0.2
NOISE_STDEV = 0.05
N_SACCADES = 11
EXIT_BETA = 0
SIM_BETA = 1
# units: maze coordinates / duration units. set to rounded avg sim speed of human data
SIM_SPEED = 10
MEM_CHANNELS = 8
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

MAZE_STIMULI_DIR = 'maze_stimuli_full_200trial_2repeat'
HUMAN_DATA_DIR = 'human_eye_data'
EXCLUDE_AMPLITUDE_UNDER = 1.5
EXCLUDE_AFTER_TARGDIST_UNDER = 3
REJSAMP_DATA_FILE = 'baseline_model_paths.pickle'

def set_task_parameters():
    return MAZE_SIZE, MAZE_DIR, NUM_LAYERS, PIXELS_PER_SQUARE, IMG_DIM

def set_gazeRNN_hyperparameters():
    return SPATIAL_BASIS, TAU, NOISE_STDEV, N_SACCADES, EXIT_BETA, SIM_BETA, SIM_SPEED, MEM_CHANNELS, BATCH_SIZE, LEARNING_RATE

def set_eval_parameters():
    return MAZE_STIMULI_DIR, HUMAN_DATA_DIR, EXCLUDE_AMPLITUDE_UNDER, EXCLUDE_AFTER_TARGDIST_UNDER, REJSAMP_DATA_FILE