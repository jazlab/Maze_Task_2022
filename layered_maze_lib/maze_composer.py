"""MAze composer class."""

import numpy as np
import os
from . import path_dataset
from scipy import signal as scipy_signal


class MazeComposer():
    """Generates random mazes composed of overlaying paths."""

    def __init__(self,
                 path_dir,
                 num_layers,
                 pixels_per_square=4):
        """Constructor.
        
        Args:
            path_dir: String. Directory of path dataset to use for composing
                mazes.
            num_layers: Int. Number of paths to compose for each maze.
            pixels_per_square: Int. Number of pixels for each maze square during
                rendering.
        """
        self._num_layers = num_layers
        self._pixels_per_square = pixels_per_square

        if pixels_per_square % 2 != 0:
            raise ValueError(
                f'pixels_per_square is {pixels_per_square} but must be even '
                'for the ball path to line up in the center of the maze path.')

        # Load mazes
        filenames = os.listdir(path_dir)
        self._mazes = []
        for k in filenames:
            new_mazes = np.load(os.path.join(path_dir, k), allow_pickle=True)
            self._mazes.extend(new_mazes)
        self._num_mazes = len(self._mazes)

    def _transform_maze(self, maze, path):
        """Randomly flip/rotate maze and path."""
        if np.random.rand() < 0.5:
            maze, path = path_dataset.flip_maze_and_path_up_down(maze, path)
        if np.random.rand() < 0.5:
            maze, path = path_dataset.rotate_maze_and_path(maze, path)
        return maze, path

    def _render_maze(self, maze):
        """Convert small binary maze array to full-size maze array.
        
        The returned maze array has values 0 for outside the path, 1 for path
        walls, and -1 for path interior. This coding is important for the maze
        composition.
        """
        maze = np.repeat(
            np.repeat(maze, self._pixels_per_square, axis=0),
            self._pixels_per_square, axis=1)
        kernel = np.ones((2, 2))
        maze = scipy_signal.convolve2d(
            maze, kernel, mode='valid', boundary='symm')
        maze[maze == 4] = -1
        maze[maze > 0] = 1
        return maze

    def _overlay_mazes(self, mazes):
        """Overlay a list of mazes with occlusion, in order."""
        final_maze = mazes[0]
        for maze in mazes[1:]:
            final_maze *= (maze == 0)
            final_maze += maze
        final_maze[final_maze < 0] = 0.
        return final_maze

    def _augment_path(self, path):
        """Convert path in units of grid square to units of pixels.
        
        The returned path is self._pixels_per_square times longer than the input
        path.
        """
        path = np.repeat(path, self._pixels_per_square, axis=0)
        kernel = np.ones((self._pixels_per_square, 1))
        kernel /= self._pixels_per_square
        path = scipy_signal.convolve2d(
            path, kernel, mode='valid', boundary='symm').astype(int)
        return path

    def __call__(self):
        """Sample a maze."""
        mazes = [
            self._mazes[np.random.randint(self._num_mazes)]
            for _ in range(self._num_layers)
        ]
        mazes = [self._transform_maze(*x) for x in mazes]
        path = mazes[-1][1]
        path = self._pixels_per_square * path
        path += int(self._pixels_per_square / 2) - 1
        path = self._augment_path(path)
        mazes = [x[0] for x in mazes]
        mazes = [self._render_maze(x) for x in mazes]
        maze = self._overlay_mazes(mazes)
        return maze, path
