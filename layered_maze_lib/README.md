# Library for generating mazes by layering paths for MazeSetGo

By Nick Watters, 2022

## Introduction

This library contains code for generating mazes. The maze generation is done in
the following way:
* Pre-generate a bunch of paths
* When generating a maze, layer a random sample of paths (with occlusion)

This has the following features:
1. Distractor paths have similar statistics to the ball path.
2. Mazes can be generated efficiently online during model-training, which
   reduces the likelihood of overfitting.

## Code Contents

There are three core library files:
* `generate_path.py` -- this generates a single path, and is used during the
  pre-generation of a paths dataset.
* `path_dataset.py` -- this has a function `generate_path_dataset()` that
  generates a large (user-specified) set of paths, as well as some helper
  functions for transforming paths. This is also used during path dataset
  pre-generation.
* `maze_composer.py` -- this has a callable class that composes paths to produce
  a sampled maze and ball path. This should be used for model-training.

In addition, there are a few short files with the `run_` prefix, but more on
those later.

The `path_datasets/` directory contains path datasets.

## Dependencies

This library requires you to install the following packages:
```
$ pip install absl-py
$ pip install matplotlib
$ pip install scipy
```

## Getting started

Start by running `run_demo_maze_composer.py`. This file should run
out-of-the-box and plot some mazes.

After running this demo, look at the `MazeComposer` construction in
`run_demo_maze_composer.py`. This takes a directory name of the path dataset
(`path_dir`) and a number of paths to layer per maze (`num_layers`). Given that
syntax, you can construct a MazeComposer class in your model training code ---
calling that class returns a maze and ball path.

If you want to generate a new dataset of paths (e.g. with more paths, a
different maze size, etc.), take a look at `run_path_dataset.py`.
