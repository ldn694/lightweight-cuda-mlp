#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "Tensor.cuh"
#include <vector>

__global__ void sgdKernel(float *param, float *grad, int size, float lr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        param[idx] -= lr * grad[idx];
    }
}

class Optimizer
{
private:
    float learningRate;

public:
    Optimizer(float lr) : learningRate(lr) {}

    void step(const std::vector<std::pair<Tensor *, Tensor *>> &paramsAndGrads)
    {
        for (auto &pg : paramsAndGrads)
        {
            Tensor *param = pg.first;
            Tensor *grad = pg.second;

            int size = param->getSize();
            dim3 block(256);
            dim3 grid((size + block.x - 1) / block.x);

            sgdKernel<<<grid, block>>>(
                param->getData(), grad->getData(), size, learningRate);
        }
    }
};

#endif // OPTIMIZER_CUH
