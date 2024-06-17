# Selective Consistency of Recurrent Neural Networks Induced by Plasticity as a Mechanism of Unsupervised Perceptual Learning
This repository contains the Python code for the paper:

Goto, Kitajo, "Selective consistency of recurrent neural networks induced by plasticity as a mechanism of unsupervised perceptual learning", 2024, PLoS Computational Biology.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)

## Introduction

This repository provides the implementation of the experiments and analyses described in the paper. The code is freely available for academic and research purposes.

## Python Environment
- Python version  
3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)]
- Version info.  
sys.version_info(major=3, minor=9, micro=13, releaselevel='final', serial=0)
- numpy==1.21.5
- cupy==11.4.0
- model_cp does not have a __version__ attribute.
- a_weight does not have a __version__ attribute.
- pandas==1.4.4

## Usage

To run the simulation, use the following command:
    ```bash
    python main.py
    ```

and then you will get "data_{pred_num}_hebb.csv" and "data_{pred_num}_nohebb.csv". 
You can see the network's output time series of hebbian and non-hebbian network, respectively in these data files.

## Code Structure

- `main.py`: Script to run the main simulation.
- `model.py`: Script for plastic recurrent neural network.
- `a_weight.py`: Script for a-weighting filter.
