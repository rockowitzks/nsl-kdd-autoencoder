# NSL KDD Autoencoder and Classifier

This repository contains code and resources to train both binary and multi-class autoencoders and classifiers using the NSL KDD dataset.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)

## Introduction

The NSL KDD dataset is an improved version of the well-known KDD Cup 1999 dataset, which has been widely used for intrusion detection system (IDS) evaluation. This codebase aims to showcase the power of autoencoders in feature learning and classification tasks for intrusion detection.

### Features:

1. Binary Classification: Distinguishes between normal and attack classes.
2. Multi-Class Classification: Distinguishes among different types of attacks.
3. Autoencoders: Used for dimensionality reduction and feature extraction.
4. Visualization tools: For understanding feature distributions and model performance.

## Setup and Installation

### Prerequisites:

- Python 3.10.10
- Keras 2.14.0
- Pandas 1.2.2
- Matplotlib 3.8.0
- Scikit-learn 1.3.2
- NumPy 1.24.1
- TensorFlow 2.14.0

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

### Dataset: 
The NSL KDD dataset can be downloaded [here](https://www.unb.ca/cic/datasets/nsl.html). After downloading, place the dataset files in the data/ directory.

## Usage

### Binary Classification:

To train and evaluate a binary classifier, run:

```bash
python nsl_kdd_auto_encoder.py --mode binary
```

### Multi-Class Classification:

To train and evaluate a multi-class classifier, run:

```bash
python nsl_kdd_auto_encoder.py --mode multi
```


