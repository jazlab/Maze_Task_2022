"""Generate a dataset of paths."""

from absl import app
import generate_path
import numpy as np
import os
import path_dataset

_MAZE_SIZE = 14  # Size of the maze
_DIR_SUFFIX = '_v0'  # Suffix for the data directory

# Maximum number of unique paths for each (start, end) pair
_SAMPLES_PER_PAIR = 100

# Number of data paths to attempt to generate. The true number of data points
# will be significantly less than this because of rejection sampling.
_NUM_TRIES = int(2e5)

# Minimum length of a segment in the path before a turn
_MIN_SEGMENT_LENGTH = 3

# Probability when generating a path of turning at each step after
# _MIN_SEGMENT_LENGTH
_TURN_PROB = 0.4


def main(_):
    # Prepare directory to write data to
    maze_size_dir = os.path.join(
        os.getcwd(), 'path_datasets', 'maze_size_' + str(_MAZE_SIZE))
    if not os.path.exists(maze_size_dir):
        os.makedirs(maze_size_dir)
    num_samples_name = (
        'samples_per_pair_' + str(_SAMPLES_PER_PAIR) + _DIR_SUFFIX)
    write_dir = os.path.join(maze_size_dir, num_samples_name)
    if os.path.exists(write_dir):
        raise ValueError(f'Directory {write_dir} already exists.')
    os.makedirs(write_dir)
    print(f'Writing data to {write_dir}')

    # Generate the mazes
    path_generator = generate_path.PathGenerator(
        maze_shape=(_MAZE_SIZE, _MAZE_SIZE),
        min_segment_length=_MIN_SEGMENT_LENGTH,
        turn_prob=_TURN_PROB,
    )
    mazes = path_dataset.generate_path_dataset(
        path_generator, num_samples=_SAMPLES_PER_PAIR, num_tries=_NUM_TRIES)

    # Write mazed to files
    for k, v in mazes.items():
        filename = os.path.join(write_dir, str(k))
        try:
            np.save(filename, v)
        except:
            os.remove(filename + '.npy')
            print(f'could not save {k}')
            continue

    print('done')



if __name__ == "__main__":
    app.run(main)
