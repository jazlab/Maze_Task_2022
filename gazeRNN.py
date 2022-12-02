import os
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pickle

from settings import set_task_parameters, set_gazeRNN_hyperparameters
from process_human_data import human_sim_speed
from layered_maze_lib import maze_composer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAZE_SIZE, MAZE_DIR, NUM_LAYERS, PIXELS_PER_SQUARE, IMG_DIM = set_task_parameters()
SPATIAL_BASIS, TAU, NOISE_STDEV, N_SACCADES, EXIT_BETA, SIM_BETA, SIM_SPEED, MEM_CHANNELS, BATCH_SIZE, LEARNING_RATE = set_gazeRNN_hyperparameters()
SPATIAL_BASIS = SPATIAL_BASIS.to(device)
SIM_SPEED = human_sim_speed()

N_TEST = 100

TRAIN_SAVES_DIR = 'train_saves'
SAVED_MODELS_DIR = 'saved_models'

version = 'version_string'

if not os.path.exists(TRAIN_SAVES_DIR):
    os.makedirs(TRAIN_SAVES_DIR)
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)


def rotate(points, theta, xo=IMG_DIM / 2 - 0.5, yo=IMG_DIM / 2 - 0.5):
    """Rotate points around xo,yo by theta (rad) clockwise."""
    xr = np.cos(theta) * (points[:, 0] - xo) - np.sin(theta) * (points[:, 1] - yo) + xo
    yr = np.sin(theta) * (points[:, 0] - xo) + np.cos(theta) * (points[:, 1] - yo) + yo
    return np.dstack((xr, yr))[0]


def generate_batch(composer, batch_size, device=device):
    mazes, entrances, exits, paths = [], [], [], []
    for _ in range(batch_size):
        maze, path = composer()
        # flip coordinates (row, col) or (y, x) to (col, row) or (x, y) before rotating randomly
        path = np.flip(path, axis=1)
        n_rot = np.random.randint(4)  # number of counterclockwise rotations
        maze = np.rot90(maze, k=n_rot)
        theta = (-n_rot * np.pi / 2) % (2 * np.pi)  # radians of rotation, in clockwise radians
        path = rotate(path, theta)
        entrance = path[0]
        exit_point = path[-1]

        mazes.append(1 - maze)  # flip 0s and 1s
        entrances.append(entrance)
        exits.append(exit_point)
        paths.append(path)
    mazes = torch.as_tensor(mazes).float().to(device)
    entrances = torch.as_tensor(entrances).float().to(device)
    exits = torch.as_tensor(exits).float().to(device)
    return mazes, entrances, exits, paths


def create_masks(spatial_basis, centers, tau):
    d = (torch.tile(spatial_basis, (centers.shape[0], 1, 1, 1)) - torch.reshape(centers, (-1, 1, 1, 2))).square().sum(dim=3).sqrt()
    return torch.exp(-d * tau)


class SaccadeCNN(nn.Module):
    def __init__(self, mem_channels, do_sim):
        super(SaccadeCNN, self).__init__()
        self.do_sim = do_sim
        self.mem_channels = mem_channels
        self.conv1 = nn.Conv2d(1 + self.mem_channels + 1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)

        self.fc1 = nn.Linear(64 * 4 ** 2, 64)
        self.eye_pos_fc1 = nn.Linear(64, 16)
        self.eye_pos_fc2 = nn.Linear(16, 16)
        self.eye_pos_fc3 = nn.Linear(16, 2)
        self.ball_pos_fc1 = nn.Linear(64, 16)
        self.ball_pos_fc2 = nn.Linear(16, 16)
        self.ball_pos_fc3 = nn.Linear(16, 2)

    def forward(self, img, memory, curr_mask, noise):
        x = torch.cat([img * curr_mask + noise * (1 - curr_mask), memory, curr_mask.detach()], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        eye_pos = F.relu(self.eye_pos_fc1(x))
        eye_pos = F.relu(self.eye_pos_fc2(eye_pos))
        eye_pos = self.eye_pos_fc3(eye_pos)

        ball_pos = None
        if self.do_sim:
            ball_pos = F.relu(self.ball_pos_fc1(x))
            ball_pos = F.relu(self.ball_pos_fc2(ball_pos))
            ball_pos = self.ball_pos_fc3(ball_pos)

        return eye_pos, ball_pos


class MemCNN(nn.Module):
    def __init__(self, mem_channels):
        super(MemCNN, self).__init__()
        self.mem_channels = mem_channels
        self.conv1 = nn.Conv2d(1 + self.mem_channels + 1, 8, 3, padding='same')
        self.conv2 = nn.Conv2d(8, 8, 3, padding='same')
        self.conv3 = nn.Conv2d(8, self.mem_channels, 3, padding='same')

    def forward(self, img, memory, curr_mask, noise):
        x = torch.cat([img * curr_mask + noise * (1 - curr_mask), memory, curr_mask.detach()], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def loss_fn(eye_positions, exits, ball_positions, true_ball_positions, sim_beta, exit_beta):
    """Return sum of simulation and exit loss weighted by beta hyperparameters."""
    sim_loss = torch.mean((ball_positions - true_ball_positions) ** 2, dim=1)
    exit_loss = torch.mean((eye_positions - exits) ** 2, dim=1)
    return torch.mean(sim_beta * sim_loss + exit_beta * exit_loss)


def true_ball_position(maze_path, sim_speed):
    """Given maze path and ball speed, returns position of true ball."""
    return tuple(np.interp(sim_speed, np.arange(maze_path[:, i].size), maze_path[:, i]) for i in range(2))


def run_batch(action, saccadecnn, memcnn, images, entrances, exits, paths, saccadecnn_optimizer=None, memcnn_optimizer=None):
    """Train or test on a single batch."""
    with torch.set_grad_enabled(action == 'train'):
        eye_positionss = []
        ball_positionss = []
        true_ball_positionss = []

        if action == 'train':
            saccadecnn_optimizer.zero_grad()
            memcnn_optimizer.zero_grad()
        loss = 0

        batch_memory = None
        curr_masks = create_masks(SPATIAL_BASIS, entrances, TAU).to(device)

        for _ in range(N_SACCADES):
            noise = torch.normal(0, NOISE_STDEV, size=(IMG_DIM, IMG_DIM)).to(device)
            # update internal memory; initialize all 8 memory channels to the masked and noised image
            if batch_memory is None:
                batch_memory = (images * curr_masks + noise * (1 - curr_masks)).unsqueeze(1).repeat(1,
                                                                                                    saccadecnn.mem_channels,
                                                                                                    1, 1).to(device)
            else:
                batch_memory = memcnn(images.unsqueeze(1), batch_memory, curr_masks.unsqueeze(1), noise)
            # find next eye position
            eye_positions, ball_positions = saccadecnn(images.unsqueeze(1), batch_memory, curr_masks.unsqueeze(1), noise)
            # make a saccade
            curr_masks = create_masks(SPATIAL_BASIS, eye_positions, TAU).to(device)
            # compute the loss
            true_ball_poss = torch.as_tensor([true_ball_position(path, SIM_SPEED) for path in paths]).to(device)
            # if no simulation, set ball_positions to eye_positions just to preserve loss tensor dimensions
            if ball_positions is None:
                ball_positions = eye_positions
            loss += loss_fn(eye_positions, exits, ball_positions, true_ball_poss, SIM_BETA, EXIT_BETA)

            eye_positionss.append(eye_positions.detach().cpu().numpy())
            ball_positionss.append(ball_positions.detach().cpu().numpy())
            true_ball_positionss.append(true_ball_poss.detach().cpu().numpy())

        # reshape network outputs
        eye_positionss = np.transpose(np.array(eye_positionss), axes=(1, 0, 2))
        ball_positionss = np.transpose(np.array(ball_positionss), axes=(1, 0, 2))
        true_ball_positionss = np.transpose(np.array(true_ball_positionss), axes=(1, 0, 2))

        if action == 'train':
            loss.backward()
            saccadecnn_optimizer.step()
            memcnn_optimizer.step()

        return loss.item(), eye_positionss, ball_positionss, true_ball_positionss


def as_minutes(seconds):
    return '{:d}m {:d}s'.format(math.floor(seconds / 60), int(seconds - 60 * math.floor(seconds / 60)))


def train(saccadecnn, memcnn, composer, version, learning_rate, batch_size, max_iters=5e6, print_every=1e3, test_every=1e4, save_every=1e4):
    train_losses, test_losses = [], []
    loss_buffer = 0

    saccadecnn_optimizer = optim.Adam(saccadecnn.parameters(), lr=learning_rate)
    memcnn_optimizer = optim.Adam(memcnn.parameters(), lr=learning_rate)

    print('Training model version: {}'.format(version))
    start_time = time.time()
    for curr_iter in range(1, int(max_iters) + 1):
        mazes, entrances, exits, paths = generate_batch(composer, batch_size)

        train_loss, _, _, _ = run_batch('train',
                                        saccadecnn, memcnn,
                                        mazes, entrances, exits, paths,
                                        saccadecnn_optimizer, memcnn_optimizer,)
        loss_buffer += train_loss
        train_losses.append(train_loss)

        if curr_iter % print_every == 0:
            loss_avg = loss_buffer / print_every
            loss_buffer = 0
            time_elapsed = as_minutes(time.time() - start_time)
            print('{} ({:d} {:d}%) {:.4f}'.format(time_elapsed, curr_iter, round(curr_iter / max_iters * 100), loss_avg))

        if curr_iter % test_every == 0:
            test_loss, _, _, _ = run_batch('test',
                                           saccadecnn, memcnn,
                                           test_imgs, test_entrances, test_exits, test_paths)
            test_losses.append(test_loss)
            print('Current Test Loss: {:.3f}'.format(test_loss))

        if curr_iter % save_every == 0:
            with open('{}/train_losses-{}.pickle'.format(TRAIN_SAVES_DIR, version), 'wb') as f:
                pickle.dump(train_losses, f)
            with open('{}/val_losses-{}.pickle'.format(TRAIN_SAVES_DIR, version), 'wb') as f:
                pickle.dump(test_losses, f)
            torch.save(saccadecnn.state_dict(), '{}/saccadecnn-{}-it{}.pt'.format(SAVED_MODELS_DIR, version, curr_iter))
            torch.save(memcnn.state_dict(), '{}/memcnn-{}-it{}.pt'.format(SAVED_MODELS_DIR, version, curr_iter))


if __name__ == '__main__':
    test_composer = maze_composer.MazeComposer(path_dir=MAZE_DIR, num_layers=NUM_LAYERS, pixels_per_square=PIXELS_PER_SQUARE)
    test_imgs, test_entrances, test_exits, test_paths = generate_batch(test_composer, N_TEST, device='cpu')

    saccadecnn = SaccadeCNN(MEM_CHANNELS, do_sim=(SIM_BETA > 0)).to(device)
    memcnn = MemCNN(MEM_CHANNELS).to(device)

    composer = maze_composer.MazeComposer(path_dir=MAZE_DIR, num_layers=NUM_LAYERS, pixels_per_square=PIXELS_PER_SQUARE)

    train(saccadecnn, memcnn, composer, version, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

