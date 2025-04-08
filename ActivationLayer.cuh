#ifndef ACTIVATION_LAYER_CUH
#define ACTIVATION_LAYER_CUH

#include "LayerBase.cuh"

__global__
void reluForwardKernel(const float* input, float* output, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__
void reluBackwardKernel(const float* input, const float* gradOut, float* gradIn, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        gradIn[idx] = (input[idx] > 0.0f) ? gradOut[idx] : 0.0f;
    }
}

class ReLULayer : public LayerBase {
private:
    Tensor* input;
    Tensor* output;
    Tensor* gradInput;

public:
    ReLULayer() : input(nullptr), output(nullptr), gradInput(nullptr) {}

    ~ReLULayer() {
        if (output)    delete output;
        if (gradInput) delete gradInput;
    }

    Tensor* forward(Tensor* inputTensor) override {
        this->input = inputTensor;
        int size = input->getSize();

        if (!output) {
            output = new Tensor(size);
        }

        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        reluForwardKernel<<<grid, block>>>(
            input->getData(),
            output->getData(),
            size
        );

        return output;
    }

    Tensor* backward(Tensor* gradOutput) override {
        int size = gradOutput->getSize();
        if (!gradInput) {
            gradInput = new Tensor(size);
        }

        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);

        reluBackwardKernel<<<grid, block>>>(
            input->getData(),
            gradOutput->getData(),
            gradInput->getData(),
            size
        );

        return gradInput;
    }

    std::vector<std::pair<Tensor*, Tensor*>> getParamsAndGrads() override {
        // ReLU has no parameters
        return {};
    }
};

#endif // ACTIVATION_LAYER_CUH
