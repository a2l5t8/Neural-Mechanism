Sure! Here's a detailed README for the GitHub repository on Spiking Neural Networks with various neural mechanisms:

---

# Spiking Neural Network Repository

Welcome to the Spiking Neural Network Repository! This repository focuses on the neural mechanisms, implementation, and analysis of Spiking Neural Networks (SNNs). The implementations leverage the CoNeX and PyMoNNtorch frameworks using PyTorch as the backend.

## Table of Contents
- [Introduction](#introduction)
- [Neural Mechanisms](#neural-mechanisms)
  - [Lateral Inhibition](#lateral-inhibition)
  - [K-Winner-Takes-All (KWTA)](#k-winner-takes-all-kwta)
  - [Homeostasis](#homeostasis)
  - [Weight Normalization](#weight-normalization)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Spiking Neural Networks (SNNs) are a class of neural networks that more closely mimic the functioning of biological neurons compared to traditional artificial neural networks. This repository includes implementations of SNNs with key neural mechanisms like Lateral Inhibition, K-Winner-Takes-All (KWTA), Homeostasis, and Weight Normalization. The code is written using the CoNeX and PyMoNNtorch frameworks, which provide powerful tools for modeling and simulating spiking neural networks on top of PyTorch.

## Neural Mechanisms

### Lateral Inhibition
Lateral Inhibition is a process where neurons inhibit their neighbors' activity to enhance the contrast of neural representations. This mechanism is crucial for edge detection and feature extraction in sensory systems.

### K-Winner-Takes-All (KWTA)
K-Winner-Takes-All is a competitive mechanism where only the top k neurons with the highest activations are allowed to remain active while the others are suppressed. This helps in sparse coding and efficient information representation.

### Homeostasis
Homeostasis refers to the regulatory mechanisms that maintain stable activity levels in neurons over time. It ensures network stability and prevents runaway excitation or inhibition.

### Weight Normalization
Weight Normalization techniques are used to adjust the synaptic weights to maintain balanced activity within the network. This helps in preventing saturation and ensures effective learning.

## Installation

To use the code in this repository, you need to have Python and PyTorch installed. Additionally, you will need to install the CoNeX and PyMoNNtorch frameworks. You can install the required dependencies using pip:

```bash
pip install torch
pip install cnrl-conex
pip install pymonntorch
```

## Usage

Each neural mechanism and SNN model is contained within its own module. You can import and use them in your own Python scripts. Below is an example of how to use the Lateral Inhibition mechanism in an SNN model:

```python
import torch
from conex import *
from pymonntorch import *

# Network Construction
net = Network(behavior=prioritize_behaviors([
    TimeResolution(),
]))

ng = NeuronGroup(net = net, size = 10, behavior={
    # 9 : InputCurrent(mean = 30, std = 15),
    10 : SimpleDendriteStructure(),
    11 : SimpleDendriteComputation(),
    12 : LIF(
        R = 1,
        threshold = -30,
        tau = 20,
        v_reset = -80,
        v_rest = -60,
    ),
    13:InherentNoise(scale = 7),
    14:Fire(),


    1000 : Recorder(["v", "I"]) 
})

# Network initilization and simulation
net.initialize()
net.simulate_iterations(1000)

```

## Examples

The `examples` are all included in notebooks which contains detailed example scripts demonstrating how to use each neural mechanism and SNN model. you can also use `CoNeX` and `Pymonntorch` examples which provides step-by-step instructions and explanations to help you get started quickly.

## Contributing

We welcome contributions! If you have improvements or new models to add, please fork the repository and submit a pull request. Make sure to include tests and documentation for your changes.
