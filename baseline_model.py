import pickle
import numpy as np

from settings import set_task_parameters, set_eval_parameters
from process_human_data import get_human_saccade_kernels, get_test_stimuli

MAZE_SIZE, MAZE_DIR, NUM_LAYERS, PIXELS_PER_SQUARE, IMG_DIM = set_task_parameters()
_, _, _, _, REJSAMP_DATA_FILE = set_eval_parameters()
MAX_ATTEMPTS_PER_MAZE = 5000

test_mazes, test_entrances, test_exits, test_paths = get_test_stimuli('cpu')
amplitude_kernel, angle_kernel, num_saccades_kernel = get_human_saccade_kernels()


def in_maze_bounds(point):
    return (-0.5 <= point[0] <= IMG_DIM-0.5) and (-0.5 <= point[1] <= IMG_DIM-0.5)

# rejection sample eye movement paths
paths = []
best_dists = []
for m_i, _ in enumerate(test_mazes):
    print(m_i + 1, end=' ')
    entrance_point = test_entrances[m_i]
    exit_point = test_exits[m_i]
    best_path, best_dist = None, 1e9
    out_of_bounds_path = False
    # for each attempt at constructing a full path
    for attempt in range(MAX_ATTEMPTS_PER_MAZE):
        path = [entrance_point]
        num_saccades = int(np.round(num_saccades_kernel.resample(1)[0][0]))
        # generate each saccade in the attempt
        for fixation in range(num_saccades - 1):
            amp = amplitude_kernel.resample(1)[0][0]
            ang = np.pi / 180 * angle_kernel.resample(1)[0][0]
            # ang is change in angle from previous saccade
            prev_ang_from_x = 0 if len(path) < 2 else np.arctan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])
            next_pos = path[-1] + amp * np.array([np.cos(ang + prev_ang_from_x), np.sin(ang + prev_ang_from_x)])
            if not in_maze_bounds(next_pos):
                out_of_bounds_path = True
                break
            path.append(next_pos)
        # skip attempts with out of bounds path
        if out_of_bounds_path:
            out_of_bounds_path = False
            continue
        dist = np.linalg.norm(path[-1] - exit_point)
        if dist < best_dist:
            best_path, best_dist = path, dist
    print(attempt + 1, np.round(best_dist, 3))
    best_dists.append(best_dist)
    paths.append(np.array(best_path))


with open(REJSAMP_DATA_FILE, 'wb') as f:
    pickle.dump(paths, f)


