#ifndef ACTIVATION_LAYER_CUH
#define ACTIVATION_LAYER_CUH

#include "Tensor.cuh"
#include "LayerBase.cuh"
#include <vector>
#include <utility>
#include <cuda_runtime.h>

// CUDA kernel for forward ReLU activation: out = max(0, x)
__global__ void reluForwardKernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

// CUDA kernel for backward ReLU: gradInput = gradOut if input > 0 else 0
__global__ void reluBackwardKernel(float* gradInput, const float* gradOut, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = input[idx] > 0 ? gradOut[idx] : 0.0f;
    }
}

class ReLULayer : public LayerBase {
private:
    Tensor* input;
    Tensor* output;
    Tensor* gradInput;

public:
    ReLULayer() : input(nullptr), output(nullptr), gradInput(nullptr) {}

    ~ReLULayer() override {
        delete output;
        delete gradInput;
    }

    Tensor* forward(Tensor* inputTensor) override {
        input = inputTensor; // save for backward
        int size = inputTensor->getSize();

        if (!output) {
            output = new Tensor(size);
        }

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        reluForwardKernel<<<numBlocks, blockSize>>>(output->getData(), inputTensor->getData(), size);
        cudaDeviceSynchronize();

        return output;
    }

    Tensor* backward(Tensor* gradOutput) override {
        int size = gradOutput->getSize();

        if (!gradInput) {
            gradInput = new Tensor(size);
        }

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        reluBackwardKernel<<<numBlocks, blockSize>>>(gradInput->getData(), gradOutput->getData(), input->getData(), size);
        cudaDeviceSynchronize();

        return gradInput;
    }

    std::vector<std::pair<Tensor*, Tensor*>> getParamsAndGrads() override {
        return {};
    }
};

#endif // ACTIVATION_LAYER_CUH
