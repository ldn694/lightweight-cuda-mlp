#ifndef LINEAR_LAYER_CUH
#define LINEAR_LAYER_CUH

#include "LayerBase.cuh"
#include <random>

__global__ void linearForwardKernel(const float *input, const float *weight, const float *bias,
                                    float *output, int batchSize, int inSize, int outSize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < batchSize * outSize)
    {
        int row = idx / outSize; // which sample in the batch
        int col = idx % outSize; // which output dimension

        float sum = bias[col]; // start with the bias for that output dimension
        for (int i = 0; i < inSize; i++)
        {
            sum += input[row * inSize + i] * weight[i * outSize + col];
        }
        output[row * outSize + col] = sum;
    }
}

__global__ void linearBackwardKernel(const float *gradOut, const float *input, const float *weight,
                                     float *gradInput, float *gradWeight, float *gradBias,
                                     int batchSize, int inSize, int outSize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // gradBias
    if (idx < outSize)
    {
        float sum = 0.0f;
        for (int b = 0; b < batchSize; b++)
        {
            sum += gradOut[b * outSize + idx];
        }
        atomicAdd(&gradBias[idx], sum);
    }

    // gradWeight
    if (idx < inSize * outSize)
    {
        int i = idx / outSize;
        int j = idx % outSize;
        float sum = 0.0f;
        for (int b = 0; b < batchSize; b++)
        {
            sum += input[b * inSize + i] * gradOut[b * outSize + j];
        }
        atomicAdd(&gradWeight[idx], sum);
    }

    // gradInput
    if (idx < batchSize * inSize)
    {
        int b = idx / inSize;
        int i = idx % inSize;
        float sum = 0.0f;
        for (int j = 0; j < outSize; j++)
        {
            sum += gradOut[b * outSize + j] * weight[i * outSize + j];
        }
        atomicAdd(&gradInput[idx], sum);
    }
}

class LinearLayer : public LayerBase
{
private:
    int inSize;
    int outSize;
    int batchSize;

    Tensor *d_weight;     // shape: [inSize, outSize]
    Tensor *d_bias;       // shape: [outSize]
    Tensor *d_gradWeight; // same shape as weight
    Tensor *d_gradBias;   // same shape as bias

    Tensor *input;     // pointer to input from previous layer
    Tensor *output;    // pointer to output
    Tensor *gradInput; // pointer to grad wrt input

public:
    LinearLayer(int inSize, int outSize, int batchSize)
        : inSize(inSize), outSize(outSize), batchSize(batchSize)
    {
        d_weight = new Tensor(inSize * outSize);
        d_bias = new Tensor(outSize);
        d_gradWeight = new Tensor(inSize * outSize);
        d_gradBias = new Tensor(outSize);
        output = nullptr;
        gradInput = nullptr;

        // Random initialization of weights, zero biases
        std::vector<float> hostWeight(inSize * outSize);
        std::vector<float> hostBias(outSize, 0.0f);

        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (auto &w : hostWeight)
        {
            w = dist(rng);
        }

        d_weight->copyFromHost(hostWeight.data(), inSize * outSize);
        d_bias->copyFromHost(hostBias.data(), outSize);
    }

    ~LinearLayer()
    {
        delete d_weight;
        delete d_bias;
        delete d_gradWeight;
        delete d_gradBias;
        if (output)
            delete output;
        if (gradInput)
            delete gradInput;
    }

    Tensor *forward(Tensor *inputTensor) override
    {
        this->input = inputTensor; // store for backward
        if (!output)
        {
            output = new Tensor(batchSize * outSize);
        }

        dim3 block(256);
        dim3 grid((batchSize * outSize + block.x - 1) / block.x);

        linearForwardKernel<<<grid, block>>>(
            inputTensor->getData(),
            d_weight->getData(),
            d_bias->getData(),
            output->getData(),
            batchSize, inSize, outSize);

        return output;
    }

    Tensor *backward(Tensor *gradOutput) override
    {
        if (!gradInput)
        {
            gradInput = new Tensor(batchSize * inSize);
        }

        // Zero out grads
        cudaMemset(d_gradWeight->getData(), 0, inSize * outSize * sizeof(float));
        cudaMemset(d_gradBias->getData(), 0, outSize * sizeof(float));
        cudaMemset(gradInput->getData(), 0, batchSize * inSize * sizeof(float));

        dim3 block(256);
        // We need a bigger grid dimension because we handle 3 tasks in the kernel:
        //   - Grad w.r.t. bias (size outSize)
        //   - Grad w.r.t. weight (size inSize * outSize)
        //   - Grad w.r.t. input (size batchSize * inSize)
        // We'll call the kernel with a large enough range to handle all.
        int maxThreads = max(outSize, max(inSize * outSize, batchSize * inSize));
        dim3 grid((maxThreads + block.x - 1) / block.x);

        linearBackwardKernel<<<grid, block>>>(
            gradOutput->getData(),
            input->getData(),
            d_weight->getData(),
            gradInput->getData(),
            d_gradWeight->getData(),
            d_gradBias->getData(),
            batchSize, inSize, outSize);

        return gradInput;
    }

    std::vector<std::pair<Tensor *, Tensor *>> getParamsAndGrads() override
    {
        // Return pairs (param, gradParam)
        return {
            {d_weight, d_gradWeight},
            {d_bias, d_gradBias}};
    }
};

#endif // LINEAR_LAYER_CUH