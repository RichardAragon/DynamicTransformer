# Dynamic Transformer with Adaptable Quantization

This repository contains the implementation of a Transformer model with dynamically adaptable quantization levels. The model can switch between binary, ternary, and full-precision quantization modes, optimizing computational efficiency while maintaining performance.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to create a flexible Transformer architecture that can dynamically adjust its quantization levels based on performance metrics. The model supports binary, ternary, and full-precision operations, providing a balance between efficiency and accuracy.

## Features

- **Dynamic Quantization**: Switch between binary, ternary, and full-precision quantization modes during training and inference.
- **Quantization Controller**: Automatically adjust quantization levels based on real-time performance metrics.
- **Flexible Architecture**: Easily extendable to incorporate additional quantization strategies or performance metrics.

## Installation

To use this repository, clone it and install the required dependencies:

```bash
git clone https://github.com/yourusername/dynamic-transformer.git
cd dynamic-transformer
pip install -r requirements.txt
```

## Usage

### 1. Define Quantization Functions

Quantization functions for binary, ternary, and full-precision modes are defined in `quantization.py`.

```python
# quantization.py

import torch

def binary_quantize(tensor):
    return torch.sign(tensor)

def ternary_quantize(tensor):
    threshold = 0.05 * tensor.abs().mean()
    tensor = torch.where(tensor > threshold, 1, tensor)
    tensor = torch.where(tensor < -threshold, -1, tensor)
    tensor = torch.where((tensor <= threshold) & (tensor >= -threshold), 0, tensor)
    return tensor

def no_quantize(tensor):
    return tensor

def quantize(tensor, mode):
    if mode == 'binary':
        return binary_quantize(tensor)
    elif mode == 'ternary':
        return ternary_quantize(tensor)
    else:
        return no_quantize(tensor)
```

### 2. Modify Transformer Layers

The Transformer layers are modified to include quantization using the `QuantizedLinear` class defined in `model.py`.

```python
# model.py

import torch.nn as nn
import torch.nn.functional as F
from quantization import quantize

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mode = 'full'  # Default mode

    def forward(self, x):
        weight = quantize(self.linear.weight, self.mode)
        bias = quantize(self.linear.bias, self.mode)
        return F.linear(x, weight, bias)

    def set_mode(self, mode):
        self.mode = mode
```

### 3. Update Transformer Architecture

The Transformer encoder layer is updated to use `QuantizedLinear` and include mode setting functionality.

```python
# model.py (continued)

class QuantizedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(QuantizedTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = QuantizedLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = QuantizedLinear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def set_mode(self, mode):
        self.linear1.set_mode(mode)
        self.linear2.set_mode(mode)
```

### 4. Implement Quantization Controllers

The `QuantizationController` manages the quantization mode based on performance metrics.

```python
# controller.py

class QuantizationController:
    def __init__(self, model):
        self.model = model

    def set_mode(self, mode):
        for layer in self.model.modules():
            if isinstance(layer, QuantizedLinear):
                layer.set_mode(mode)

    def adjust_mode(self, performance_metric):
        if performance_metric < 0.5:
            self.set_mode('full')
        elif performance_metric < 0.8:
            self.set_mode('ternary')
        else:
            self.set_mode('binary')
```

### 5. Train and Evaluate the Model

Train the model and dynamically adjust quantization levels during training.

```python
# train.py

import torch.optim as optim
from model import QuantizedTransformerEncoderLayer
from controller import QuantizationController
import torch.nn as nn

# Define model, criterion, optimizer
model = QuantizedTransformerEncoderLayer(d_model=512, nhead=8)
controller = QuantizationController(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop
for epoch in range(10):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Adjust quantization mode based on a dummy performance metric
        performance_metric = loss.item()  # Example metric
        controller.adjust_mode(performance_metric)

    print(f"Epoch {epoch} completed. Current quantization mode: {model.linear1.mode}")
```

## Example

Run the example training script to see the dynamic Transformer model in action:

```bash
python train.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

What Exactly Does the Dynamic Transformer Do?
The Dynamic Transformer is an innovative architecture designed to optimize the efficiency and performance of Transformer models by dynamically adjusting the precision of its operations. Traditional Transformer models use full-precision (32-bit floating point) calculations, which can be computationally expensive and power-hungry. The Dynamic Transformer addresses this by introducing the ability to switch between different quantization levels—binary, ternary, and full-precision—during both training and inference.

Key Concepts
Quantization Levels:

Binary Quantization: Reduces weights and activations to two possible values (e.g., -1 and 1), drastically reducing memory usage and computational complexity.
Ternary Quantization: Uses three possible values (e.g., -1, 0, and 1), offering a balance between efficiency and expressiveness.
Full-Precision: Retains the original 32-bit floating-point values for maximum accuracy.
Dynamic Adjustment:

The model can switch between these quantization levels in real-time, based on the complexity of the input data or the computational resources available. This is akin to a car switching between 2-wheel drive and 4-wheel drive depending on the terrain.
Quantization Controller:

A central component that monitors performance metrics (such as loss, accuracy, or computational load) and dynamically adjusts the quantization mode of different parts of the model. This ensures that the model maintains high performance while optimizing for resource efficiency.
Benefits
Efficiency: By utilizing lower precision calculations (binary or ternary) where possible, the model reduces computational load and power consumption.
Flexibility: The ability to dynamically switch quantization levels allows the model to adapt to different scenarios, providing a balance between speed and accuracy as needed.
Scalability: This approach can be particularly beneficial for deployment on edge devices and environments with limited computational resources, such as mobile devices or IoT devices.
How It Works
Quantization Functions:

The model includes functions to convert weights and activations to binary, ternary, or full-precision formats. These functions are applied dynamically based on the current mode.
Modified Transformer Layers:

Core components of the Transformer model, such as linear layers, are modified to support dynamic quantization. These layers can switch between different quantization modes during operation.
Real-Time Control:

During training or inference, a quantization controller adjusts the quantization levels based on predefined performance metrics. For example, if the model detects that it is operating in a low-complexity scenario, it might switch to binary quantization to save resources.
Practical Applications
Edge Computing: Deploying AI models on edge devices where computational resources are limited.
Mobile AI: Enhancing the performance and battery life of AI applications on mobile devices.
Data Center Optimization: Reducing the energy consumption and improving the efficiency of large-scale AI deployments in data centers.
By dynamically adjusting quantization levels, the Dynamic Transformer model provides a versatile and efficient solution for a wide range of AI applications, making it a powerful tool for both researchers and practitioners.
