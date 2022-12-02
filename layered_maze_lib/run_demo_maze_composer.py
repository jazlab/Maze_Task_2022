"""Demo maze sampling."""

from absl import app
from matplotlib import pyplot as plt
import maze_composer
import os

_NUM_LAYERS = 4  # Number of paths to layer to make a maze
_SQRT_SAMPLES = 4  # Square root of the number of maze samples to display
_PIXELS_PER_SQUARE = 4  # Pixels per maze grid square. Should be an even number.
_RENDER_PATH = True  # Whether to render the ball path in gray.


def main(_):
    path_dir = os.path.join(
        os.getcwd(), 'path_datasets', 'maze_size_14', 'samples_per_pair_100_v0')
    composer = maze_composer.MazeComposer(
        path_dir=path_dir,
        num_layers=_NUM_LAYERS,
        pixels_per_square=_PIXELS_PER_SQUARE,
    )

    fig, axes = plt.subplots(_SQRT_SAMPLES, _SQRT_SAMPLES, figsize=(9, 9))
    for i in range(_SQRT_SAMPLES):
        for j in range(_SQRT_SAMPLES):
            maze, path = composer()
            if _RENDER_PATH:
                maze[path[:, 0], path[:, 1]] = 0.5
            axes[i, j].imshow(maze, cmap='gray')
            axes[i, j].axis('off')
    print(f'Maze shape: {maze.shape}')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    app.run(main)
