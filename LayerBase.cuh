#ifndef LAYER_BASE_CUH
#define LAYER_BASE_CUH

#include "Tensor.cuh"
#include <vector>

class LayerBase
{
public:
    virtual ~LayerBase() {}

    // Forward pass: input -> output
    virtual Tensor *forward(Tensor *input) = 0;

    // Backward pass: gradOutput -> gradInput
    virtual Tensor *backward(Tensor *gradOutput) = 0;

    // Return all parameters (weights/biases) so an optimizer can update them
    // Return format: [param1, gradParam1, param2, gradParam2, ...]
    virtual std::vector<std::pair<Tensor *, Tensor *>> getParamsAndGrads() = 0;
};

#endif // LAYER_BASE_CUH