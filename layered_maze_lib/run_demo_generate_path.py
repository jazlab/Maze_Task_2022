"""Demo path generation."""

from absl import app
import generate_path
from matplotlib import pyplot as plt

_NUM_SAMPLES = 10  # Number of paths to generate
_MAZE_SHAPE = (14, 14)  # Shape of the maze in which the path lies

# Minimum length of a segment in the path before a turn
_MIN_SEGMENT_LENGTH = 3

# Probability when generating a path of turning at each step after
# _MIN_SEGMENT_LENGTH
_TURN_PROB = 0.4


def main(_):
    path_generator = generate_path.PathGenerator(
        maze_shape=(10, 10),
        min_segment_length=3,
        turn_prob=0.4,
    )
    for _ in range(_NUM_SAMPLES):
        maze, _ = path_generator()
        plt.figure()
        plt.imshow(maze)
    plt.show()


if __name__ == "__main__":
    app.run(main)
