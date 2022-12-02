# Data and Code for Maze-Solving Task

This repo contains data and code for the following paper:

 our maze-solving project (Jason Li, Nick
Watters, Yingting Wang, Hansem Sohn, & Mehrdad Jazayeri, 2022).


## Dependencies

This library requires you to install the following packages:
```
$ pip install matplotlib
$ pip install scipy
$ pip install torch
$ pip install pandas
$ pip install seaborn
```

Currently the model-training code `gazeRNN.py` does not run as is. There appear
to be some incorrect file paths introduced by changing repositories. The
plotting code in `compare.py` and `metric_plots.py` also raise errors. The
stimulus generation code in `layered_maze_lib` does run.
