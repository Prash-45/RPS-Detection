# ResNet-18: Transfer Learning & Fine-Tuning

## Overview

This Repo explores **ResNet-18 using Transfer Learning and Fine-Tuning** for a three-class image classification task.

The primary objective is to understand how a pretrained deep convolutional neural network can be adapted to a custom dataset while reducing the computational cost of training from scratch.

The implementation uses PyTorch and pretrained ImageNet weights, with controlled parameter freezing and selective unfreezing to study different training strategies.

## Architecture

### LeNet-5 Architecture (Baseline)

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/4cc87df5-0534-48fe-a297-10b10ac11c6b" />

ResNet introduces **Residual Learning**, which allows information to bypass intermediate layers through shortcut connections.

### ResNet-18 Architecture

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/cc008aa3-cdbc-4ea8-8c8e-3359b3e04d43" />

## Implementation Strategy

The project explores two transfer learning strategies:

**1. Feature Extraction**

- Load ResNet-18 with pretrained ImageNet weights.
- Replace the original classification layer (1000 outputs) with a new fully connected layer (3 outputs).
- Freeze all pretrained parameters.
- Train only the newly initialized classification layer.

**2. Fine-Tuning**

- Start with the same pretrained ResNet-18 architecture.
- Freeze the earlier feature extraction layers.
- Unfreeze the final residual stage (`layer4`).
- Train `layer4` and the new classification head using separate learning rates.

This allows the network to retain general visual features while adapting its deeper representations to the target dataset.

## Experimental Configuration

| Component | Configuration |
|---|---|
| Framework | PyTorch |
| Architecture | ResNet-18 |
| Pretrained weights | ImageNet |
| Input resolution | 224 × 224 |
| Output classes | 3 |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Batch size | 32 |
| Feature extraction epochs | 5 |
| Fine-tuning epochs | 5 |

## Training & Evaluation

The model is evaluated using:

- Training and validation loss
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/97e4d620-0b3b-4005-a1c0-922b3bd4f184" />

- Training and validation accuracy
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/cf1510f9-564d-44ba-850b-5b62c3a836a0" />

- Test-set classification report
<img width="1178" height="704" alt="image" src="https://github.com/user-attachments/assets/ed60f47c-491e-4ae7-81bf-2284f453928e" />

- Confusion matrix
<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/ced59437-7a33-423e-a1ef-a8e814dfca53" />


## Key Engineering Concepts

- Residual connections and deep feature learning
- Transfer learning with pretrained weights
- Selective layer freezing and unfreezing
- Discriminative learning rates
- Model parameter analysis (total vs. trainable parameters)
- Multi-class classification and performance evaluation

This notebook provides a practical exploration of residual learning and transfer learning.

By implementing both feature extraction and selective fine-tuning, the experiment demonstrates how pretrained representations can be reused and adapted for a new classification problem.

The focus is not just on achieving classification accuracy, but also on understanding **how architectural design and parameter optimization influence deep learning performance.**
