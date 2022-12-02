"""Path generator class."""

import numpy as np
from scipy import signal as scipy_signal

_MAX_TRIES = int(1e4)


class PathGenerator():
    """Callable class that generated a random path."""

    def __init__(self,
                 maze_shape,
                 min_segment_length=3,
                 turn_prob=0.2,
                 allow_neighbors=False):
        """Constructor.
        
        Args:
            maze_shape: Int tuple, (width, height).
            min_segment_length: Int.
            turn_prob: Float in [0, 1].
            allow_neighbors: Whether to allow the path to become adjacent to
                itself. The default is False, because the maze composer cannot
                handle a path adjacent to itself. A more complicated
                implementation of the maze composer could, but that would slow
                down the maze composer.
        """
        self._maze_shape = maze_shape
        self._min_segment_length = min_segment_length
        self._turn_prob = turn_prob
        self._allow_neighbors = allow_neighbors

        self._create_edgepoints()

    def _create_edgepoints(self):
        """Create list of edgepoints of the maze, excluding corners.
        
        Each element of the list is (i, j), where 1 <= i < maze_width - 1 is the
        x-coordinate and 1 <= j < maze_height - 1 is the y-coordinate.
        """
        w, h = self._maze_shape
        edgepoints = (
            [(0, i) for i in range(1, h - 1)] +
            [(w - 1, i) for i in range(1, h - 1)] +
            [(i, 0) for i in range(1, w - 1)] +
            [(i, h - 1) for i in range(1, w - 1)]
        )
        self._edgepoints = np.array(edgepoints)

    def _reset(self):
        # Sample start position
        start_pos = np.copy(
            self._edgepoints[np.random.randint(len(self._edgepoints))])

        # Get start direction
        if start_pos[0] == 0:
            start_dir = np.array([1, 0])
        elif start_pos[1] == 0:
            start_dir = np.array([0, 1])
        elif start_pos[0] == self._maze_shape[0] - 1:
            start_dir = np.array([-1, 0])
        elif start_pos[1] == self._maze_shape[1] - 1:
            start_dir = np.array([0, -1])
        else:
            raise ValueError(
                f'Invalid start_pos {start_pos}, something went wrong.')

        # Set state
        self._path = []
        self._tail_position = start_pos
        self._tail_direction = start_dir
        self._since_turn = 0
        self._maze_array = np.zeros(self._maze_shape, dtype=int)
        self._add_tail_pos()
        self._update_maze()
    
    def _add_tail_pos(self):
        self._path.append(np.copy(self._tail_position))

    def _finished(self):
        """Whether path generation is done, i.e. when tail reaches an edge."""
        diff = self._tail_position - self._edgepoints
        diff = np.sum(np.abs(diff), axis=1)
        on_edge = np.any(diff == 0)
        finished = on_edge and len(self._path) > 1
        return finished

    def _sample_turn(self):
        new_direction = np.random.choice([-1, 1], size=(2,), replace=True)
        new_direction *= 1 - np.abs(self._tail_direction)
        return new_direction.astype(int)

    def _update_maze(self):
        # Check to make sure path isn't intersecting itself
        if self._maze_array[self._tail_position[0], self._tail_position[1]] > 0:
            return False
        else:
            self._maze_array[self._tail_position[0], self._tail_position[1]] = 1
            return True

    def _step(self):
        """Step the path generation, updating the maze and tail."""
        self._since_turn += 1
        if self._since_turn >= self._min_segment_length:
            if np.random.rand() < self._turn_prob:
                self._tail_direction = self._sample_turn()
                self._since_turn = 0
        self._tail_position += self._tail_direction
        self._add_tail_pos()
        valid_update = self._update_maze()
        return valid_update

    def _try_to_generate_path(self):
        """Try to generate a path, returning whether successful."""
        self._reset()
        while not self._finished():
            valid_step = self._step()
            if not valid_step:
                return False
        self._path = np.array(self._path)

        if not self._allow_neighbors:
            # Reject if the maze loops back on itself
            self._maze_array
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            conv = scipy_signal.convolve2d(
                self._maze_array, kernel, mode='valid', boundary='symm')
            if np.sum(conv > 3) > 0:
                return False

        return True

    def __call__(self):
        """Generate a path."""
        for _ in range(_MAX_TRIES):
            valid_path = self._try_to_generate_path()
            if valid_path:
                return self._maze_array, self._path

        raise ValueError('Could not generate a prey path.')

    @property
    def maze_shape(self):
        return self._maze_shape

    @property
    def edgepoints(self):
        return self._edgepoints
