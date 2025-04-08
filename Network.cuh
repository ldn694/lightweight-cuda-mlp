#ifndef NETWORK_CUH
#define NETWORK_CUH

#include "LayerBase.cuh"
#include <vector>
#include <memory>

class Network
{
private:
    std::vector<std::unique_ptr<LayerBase>> layers;

public:
    Network() {}

    void addLayer(LayerBase *layer)
    {
        layers.emplace_back(layer);
    }

    // Forward pass through all layers
    Tensor *forward(Tensor *input)
    {
        // std::cout << "Forward pass through network" << std::endl;
        Tensor *x = input;
        for (auto &layer : layers)
        {
            x = layer->forward(x);
        }
        return x;
    }

    // Backward pass from final layer
    Tensor *backward(Tensor *gradOutput)
    {
        Tensor *grad = gradOutput;
        // traverse layers in reverse
        for (int i = (int)layers.size() - 1; i >= 0; i--)
        {
            grad = layers[i]->backward(grad);
        }
        return grad;
    }

    // Aggregate all parameters and gradients from each layer
    std::vector<std::pair<Tensor *, Tensor *>> getAllParamsAndGrads()
    {
        std::vector<std::pair<Tensor *, Tensor *>> all;
        for (auto &layer : layers)
        {
            auto pg = layer->getParamsAndGrads();
            all.insert(all.end(), pg.begin(), pg.end());
        }
        return all;
    }
};

#endif // NETWORK_CUH
