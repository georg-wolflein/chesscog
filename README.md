# chesscog

![build](https://github.com/georg-wolflein/chesscog/workflows/build/badge.svg)

_chesscog_ combines traditional computer vision techniques with deep learning to recognise chess positions from photos.

This repository contains the official code for the paper:

> [**Determining Chess Game State From an Image**](https://doi.org/10.3390/jimaging7060094)  
> Georg Wölflein and Ognjen Arandjelović  
> _Journal of Imaging_, vol. 7, no. 6, p. 94, June 2021.

I originally developed this project as part of my [master thesis](https://github.com/georg-wolflein/chesscog-report/raw/master/report.pdf) at the University of St Andrews. Documentation is available [here](https://georg-wolflein.github.io/chesscog).

## Related repositories

- [chesscog-report](https://github.com/georg-wolflein/chesscog-report): the master thesis
- [chesscog-app](https://github.com/georg-wolflein/chesscog-app): the proof-of-concept web app
- [recap](https://github.com/georg-wolflein/recap): configuration management system developed as part of this project

## Demo

See it in action at [chesscog.com](https://www.chesscog.com)!
![Screenshot](https://github.com/georg-wolflein/chesscog/raw/master/docs/demo_screenshot.png)

## Background

A casual over-the-board game between two friends will often reach an interesting position. After the game, the players will want to analyse that position on a computer, so they take a photo of the position. On the computer, they need to drag and drop pieces onto a virtual chessboard until the position matches the one they had on the photograph, and then they must double-check that they did not miss any pieces.

The goal of this project is to develop a system that is able to map a photo of a chess position to a structured format that can be understood by chess engines, such as the widely-used Forsyth–Edwards Notation (FEN).

## Overview

The chess recognition system is trained using a dataset of ~5,000 synthetically generated images of chess positions (3D renderings of different chess positions at various camera angles and lighting conditions).
The dataset is available [here](https://doi.org/10.17605/OSF.IO/XF3KA).
At a high level, the recognition system itself consists of the following pipeline:

1. board localisation (square and corner detection)
2. occupancy classification
3. piece classification
4. post-processing to generate the FEN string

## Installing

Please consult Appendix C of my [master thesis](https://github.com/georg-wolflein/chesscog-report/raw/master/report.pdf) for a detailed set of instructions pertaining to the installation and usage of _chesscog_.

There are three methods of installing and running chesscog.

1. **Using poetry (recommended).**
   Ensure you have [poetry](https://python-poetry.org) installed, then clone this repository, and install the _chesscog_:
   ```bash
   git clone https://github.com/georgw777/ chesscog.git
   cd chesscog
   poetry install
   ```
   Note that you need to run `poetry shell` to activate the virtual environment in your shell before running any of the commands later in this README.
2. **Using pip.**
   This option will install _chesscog_ locally on your machine using pip (without a virtual environment).
   ```bash
   git clone https://github.com/georgw777/ chesscog.git
   cd chesscog
   pip install .
   ```
3. **Using Docker.**
   Two Dockerfiles are provided: one for CPU (_cpu.Dockerfile_) and another with enabled GPU-acceleration (_Dockerfile_). Simply subsitute the name in the following command.
   First, build the image:

   ```bash
   docker build -t chesscog -f cpu.Dockerfile .
   ```

   Then, run the image using:

   ```bash
   docker run -it -p 8888:8888 -p 9999:9999 chesscog
   ```

   Open a browser to [http://localhost:8888](http://localhost:8888) which will display Jupyter lab running in the Docker container (the password is `chesscog`). Simply open a terminal in Jupyter lab and run the remaining instructions in this README.

### Downloading the dataset and models

To download and split the dataset, run:

```bash
python -m chesscog.data_synthesis.download_dataset
python -m chesscog.data_synthesis.split_dataset
```

Finally, ensure that you download the trained models:

```bash
python -m chesscog.occupancy_classifier.download_model
python -m chesscog.piece_classifier.download_model
```

## Command line usage

As detailed in Appendix C.2 of my [master thesis](https://github.com/georg-wolflein/chesscog-report/raw/master/report.pdf), _chesscog_ provides various scripts that can be executed from the command line.

One particularly useful one is to perform an inference (see Appendix C.2.4) which can be carried out using:

```bash
python -m chesscog.recognition.recognition path_to_image.png --white
```

The output will look as follows:

```
$ python -m chesscog.recognition.recognition data://render/train/3828.png --white
. K R . . R . .
P . P P Q . . P
. P B B . . . .
. . . . . P . .
. . b . . p . q
. p . . . . . .
p b p p . . . p
. k r . . . r .

You can view this position at https://lichess.org/editor/1KR2R2/P1PPQ2P/1PBB4/5P2/2b2p1q/1p6/pbpp3p/1kr3r1
```

Other scripts are available for tasks such as:

- training the models
- fine-tuning on custom datasets
- performing inference on fine-tuned models
- locating the corner points
- performing automated tests

Relevant documentation is available in Appendix C.2 of my [master thesis](https://github.com/georg-wolflein/chesscog-report/raw/master/report.pdf).
In particular, Appendix C.2.6 contains instructions for fine-tuning the model to your own chess set, using only two input images. 
Be sure to run `python -m chesscog.transfer_learning.create_dataset` before training (a step I forgot to mention in those instructions, see [this issue](https://github.com/georg-wolflein/chesscog/issues/15).

To see an example of how the Python API is used in practice, check out the REST API developed for the _chesscog-app_ [here](https://github.com/georg-wolflein/chesscog-app/tree/master/api).

## Citation

If you find this work helpful, please consider citing:

```
@article{wolflein2021jimaging,
  author         = {W\"{o}lflein, Georg and Arandjelovi\'{c}, Ognjen},
  title          = {Determining Chess Game State from an Image},
  journal        = {Journal of Imaging},
  volume         = {7},
  year           = {2021},
  number         = {6},
  article-number = {94}
}
```
