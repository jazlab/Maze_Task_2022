# Data and Code for Maze-Solving Task

This repo contains data and code for the following paper:

Modeling Human Eye Movements with Neural Networks in a Maze-Solving Task
(Jason Li, Nicholas Watters, Sandy Wang, Hansem Sohn, Mehrdad Jazayeri, 2022)

## Dependencies and Usage

This library requires you to install the following packages:
```
$ pip install matplotlib
$ pip install scipy
$ pip install torch
$ pip install pandas
$ pip install seaborn
```

The model-training code is located in `gazeRN.py`. This file can be run with
the command `$ python gazeRNN.py`.
The plotting code for post-training analysis is found in `compare.py` and `metric_plots.py`. 
The stimulus generation code is found in `layered_maze_lib`.
