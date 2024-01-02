# State Space LSTM

Implementation of the Topical State Space LSTM model for text sequence analysis, as described in the paper ["State Space LSTM Models with Particle MCMC Inference"](https://arxiv.org/abs/1711.11179) by Xun Zheng et al.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Results](#results)
  - [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This repository contains the implementation of the Topical State Space LSTM model, which combines the interpretability of state space models with the power of LSTMs for text sequence analysis. The model introduces topics into the LSTM framework, allowing for improved understanding of latent structures in sequential data.

## Features

- **Topical State Space LSTM Model**: Implementation of the model proposed in the paper.
- **Efficient Gibbs Inference**: Utilizes Sequential Monte Carlo (SMC) method for joint posterior sampling.
- **NLP Applications**: Adaptable for various natural language processing (NLP) tasks.

## Getting Started

### Prerequisites

- Python (>=3.6)
- Other dependencies (specified in `requirements.txt`)

### Installation

```bash
git clone https://github.com/yanisrem/SSM-Project
cd src
pip install -r requirements.txt
```

NOTE: run `load_data.ipynb` on Colab

## Results

### Dataset

IMDB dataset having 50K movie reviews for NLP or text analytics.
For more dataset information, please go through the following [link](http://ai.stanford.edu/~amaas/data/sentiment/)
