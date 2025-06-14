# Lightweight CUDA MLP

This is a minimal machine learning framework written from scratch in C++/CUDA to train a simple MLP on the MNIST dataset.

## Structure

- `main.cu`: Main entry point (training loop).
- `Tensor.cuh`: Handles GPU data storage and basic operations.
- `LayerBase.cuh`: Abstract base for all layers.
- `LinearLayer.cuh`: Fully-connected layer logic (forward/backward).
- `ActivationLayer.cuh`: Activation functions (ReLU, etc.).
- `Network.cuh`: Assembles layers, forward/backward through them in sequence.
- `Optimizer.cuh`: Optimizer (SGD) for parameter updates.
- `Dataset.cuh`: Loads MNIST dataset from the `data/` folder.

## Building

1. Ensure you have [CMake](https://cmake.org/) and NVIDIA CUDA Toolkit installed.
2. Place MNIST data (files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`) into `./data/`.
3. From the `lightweight-cuda-mlp` folder, run:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
4. If compilation is successful, an executable named lightweight_cudamlp should appear.

## Running
Execute:

```bash
./lightweight_cudamlp
```
By default, the program will train an MLP on MNIST and print out training progress and test accuracy at the end.


