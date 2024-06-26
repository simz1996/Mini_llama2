<img width="426" alt="Screen Shot 2024-06-27 at 3 12 19 AM" src="https://github.com/simz1996/Mini_llama2/assets/161559950/41103289-6e16-4d08-9914-d84e89861221">
# LLama2 Model

This repository contains an experimental implementation of the LLama2 model, inspired by Umar Jamil's lectures and based on the same model architecture used by Meta. The model is not production-ready and is intended for educational and experimental purposes only.

## Acknowledgements

Special thanks to Umar Jamil for his insightful lectures which inspired the development of this model.

## Overview

This project implements the LLama model using PyTorch. The model includes various components such as self-attention, feed-forward neural networks, and rotary embeddings.

## Features

- **Model Arguments**: Configurable parameters for the model dimensions, number of layers, heads, and more.
- **RMSNorm**: A normalization layer used in the model.
- **Self-Attention**: Implementation of the self-attention mechanism.
- **Feed-Forward Neural Network**: A feed-forward layer used in each encoder block.
- **Rotary Embeddings**: Precomputed positional frequencies for rotary embeddings.
- **Encoder Blocks**: Stacked layers of self-attention and feed-forward neural networks.
- **Transformer**: The main model class that integrates all components.

## Model Components

### ModelArgs

Defines the arguments for the model including dimensions, number of layers, and other hyperparameters.

### RMSNorm

A custom normalization layer used before the attention and feed-forward blocks.

### SelfAttention

The self-attention mechanism that processes input sequences.

### FeedForward

A feed-forward neural network used within each encoder block.

### EncoderBlock

Combines self-attention and feed-forward layers, with normalization applied before each.

### Transformer

The main class that integrates all components to form the complete LLama model.

## Installation

To install the required dependencies, run:
```bash
pip install torch

USAGE
from model import ModelArgs, Transformer
import torch

# Define model arguments
args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=50000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize the model
model = Transformer(args)

# Example input
tokens = torch.randint(0, args.vocab_size, (1, 1)).to(args.device)
start_pos = 0

# Forward pass
output = model(tokens, start_pos)
print(output)

Note
This model is experimental and not intended for production use. It is meant for educational and research purposes to understand the workings of Transformer models, specifically the LLama model used by Meta.

License
This project is license free.
