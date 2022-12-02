"""Path dataset generator.

The main function in this file is generate_path_dataset().
"""

import numpy as np


def rotate_maze_and_path(maze, path):
    """Rotate maze and path so that the entrance point is on the top.
    
    This is used for the path generation script because we only want to generate
    top-entering mazes during path dataset generation. The maze composer will
    re-rotate the paths. This strategy helps generate unique paths more
    efficiently.

    Args:
        maze: Binary array of size [height, width].
        path: Binary array of size [path_length, 2].

    Returns:
        maze: Binary array of size [height, width].
        path: Binary array of size [path_length, 2], where the first entry is at
            the top edge of the maze, i.e. [0, _].
    """
    start_orig = path[0]
    if start_orig[0] == 0:
        return maze, path
    else:
        # Rotate 90 degrees and try again
        maze = np.rot90(maze)
        maze_width = maze.shape[1]
        path = np.stack([maze_width - path[:, 1] - 1, path[:, 0]], axis=1)
        return rotate_maze_and_path(maze, path)


def flip_maze_and_path_left_right(maze, path):
    """Flip maze and path left-to-right."""
    maze = np.fliplr(maze)
    new_path = np.copy(path)
    new_path[:, 1] = maze.shape[0] - 1 - new_path[:, 1]
    return maze, new_path


def flip_maze_and_path_up_down(maze, path):
    """Flip maze and path top-to-bottom."""
    maze = np.flipud(maze)
    new_path = np.copy(path)
    new_path[:, 0] = maze.shape[0] - 1 - new_path[:, 0]
    return maze, new_path


def generate_path_dataset(path_generator, num_samples, num_tries):
    """Generate a dataset of paths.
    
    Args:
        path_generator. Instance of generate_path.PathGenerator().
        num_samples: Int. Maximum number of unique paths per [start, end] pair.
        num_tries: Int. Number of attempts at path generation. Because of
            rejection sampling the ultimate number of paths in the dataset will
            be significantly less than this.

    Returns:
        mazes: Dict. Keys are (start, end) tuples of the form
            ((start_i, start_j), (end_i, end_j)). Values are lists of
            (maze, path) tuples.
    """
    maze_shape = path_generator.maze_shape
    edgepoints = path_generator.edgepoints

    if maze_shape[0] != maze_shape[1]:
        raise ValueError('Maze must be a square.')

    startpoints = [(0, i) for i in range(1, maze_shape[0] - 1)]
    endpoints = [tuple(x) for x in edgepoints]
    pairs = [(x, y) for x in startpoints for y in endpoints if x != y]
    mazes = {pair: [] for pair in pairs}

    for i in range(num_tries):
        if i % 10000 == 0:
            print(f'Step {i} of {num_tries}')
        
        if np.all([len(x) >= num_samples for x in mazes.values()]):
            # Finished generating desired number of samples
            break

        maze, path = path_generator()

        # Rotate maze and path to have apropriate startpoint
        maze, path = rotate_maze_and_path(maze, path)

        # Add maze to mazes
        pair = (tuple(path[0]), tuple(path[-1]))
        if len(mazes[pair]) >= num_samples:
            # Have enough samples for this pair
            continue
        elif np.any([np.array_equal(maze, x[0]) for x in mazes[pair]]):
            # Duplicate maze, so skip
            continue
        else:
            # Add maze and flip of maze to mazes
            mazes[pair].append((np.copy(maze), np.copy(path)))

            flip_maze, flip_path = flip_maze_and_path_left_right(maze, path)
            if not np.array_equal(maze, flip_maze):
                flip_pair = (tuple(flip_path[0]), tuple(flip_path[-1]))
                mazes[flip_pair].append(
                    (np.copy(flip_maze), np.copy(flip_path)))
    
    if np.all([len(x) >= num_samples for x in mazes.values()]):
        print('done')
    else:
        print('Incomplete')
        num_complete = sum([len(x) >= num_samples for x in mazes.values()])
        print(f'Done {num_complete} of {len(pairs)}')
        incomplete = [k for k in mazes.keys() if len(mazes[k]) < num_samples]
        print(incomplete)

    return mazes
