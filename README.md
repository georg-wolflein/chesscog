# chesscog

![build](https://github.com/georgw777/chesscog/workflows/build/badge.svg)

_chesscog_ combines traditional computer vision techniques with deep learning to recognise chess positions from photos.

I am developing this project as part of my master thesis at the University of St Andrews.

## Related repositories

- [chesscog-report](https://github.com/georgw777/chesscog-report): the master thesis, currently a work in progress
- [chesscog-app](https://github.com/georgw777/chesscog-app): the proof-of-concept web app
- [recap](https://github.com/georgw777/recap): configuration management system developed as part of this project

## Demo

See it in action at [chesscog.com](https://www.chesscog.com)!
![Screenshot](https://github.com/georgw777/chesscog/raw/master/docs/demo_screenshot.png)

## Background

A casual over-the-board game between two friends will often reach an interesting position. After the game, the players will want to analyse that position on a computer, so they take a photo of the position. On the computer, they need to drag and drop pieces onto a virtual chessboard until the position matches the one they had on the photograph, and then they must double-check that they did not miss any pieces.

The goal of this project is to develop a system that is able to map a photo of a chess position to a structured format that can be understood by chess engines, such as the widely-used Forsyth–Edwards Notation (FEN).

## Overview

The chess recognition system is trained using a dataset of ~5,000 synthetically generated images of chess positions (3D renderings of different chess positions at various camera angles and lighting conditions).
At a high level, the recognition system itself consists of the following pipeline:

1. board localisation (square and corner detection)
2. occupancy classification
3. piece classification
4. post-processing to generate the FEN string
