"""Run timing benchmark for maze composer."""

from absl import app
import maze_composer
from matplotlib import pyplot as plt
import os
import time

_NUM_SAMPLES = 1000  # Number of mazes to generate for time estimation
_NUM_LAYERS = 20  # Number of paths to layer for composing each maze


def main(_):
    path_dir = os.path.join(
        os.getcwd(), 'path_datasets', 'maze_size_14', 'samples_per_pair_100_v0')
    composer = maze_composer.MazeComposer(
        path_dir=path_dir,
        num_layers=_NUM_LAYERS,
    )

    start_time = time.time()
    for _ in range(_NUM_SAMPLES):
        maze, _ = composer()
    end_time = time.time()
    time_per_sample = (end_time - start_time) / _NUM_SAMPLES
    print(f'Time per sample: {time_per_sample}')
    print(f'Time per 1000 samples: {1000. * time_per_sample}')


if __name__ == "__main__":
    app.run(main)
