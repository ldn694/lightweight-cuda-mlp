#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <chrono>

#include "Network.cuh"
#include "LinearLayer.cuh"
#include "ActivationLayer.cuh"
#include "Optimizer.cuh"
#include "Dataset.cuh"
#include "Tensor.cuh"

// A simple kernel to compute cross-entropy loss and its gradient w.r.t. predictions
__global__ void computeLossAndGrad(const float *preds, const float *labels, float *gradOut,
                                   float *lossOut, int batchSize, int numClasses)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < batchSize)
    {
        // label is an integer from 0..9 in this example
        int label = (int)labels[idx];
        float loss = 0.0f;

        // cross-entropy (softmax not explicitly done, this is a naive example)
        // We'll treat preds as unnormalized logits. In real code, you'd do a stable softmax.
        float maxLogit = -999999.0f;
        for (int c = 0; c < numClasses; c++)
        {
            float val = preds[idx * numClasses + c];
            if (val > maxLogit)
                maxLogit = val;
        }

        // compute normalization
        float sumExp = 0.0f;
        for (int c = 0; c < numClasses; c++)
        {
            sumExp += expf(preds[idx * numClasses + c] - maxLogit);
        }

        // compute log-softmax and loss
        for (int c = 0; c < numClasses; c++)
        {
            float logProb = (preds[idx * numClasses + c] - maxLogit) - logf(sumExp);
            float prob = expf(logProb);

            // Cross-entropy loss for the correct class
            if (c == label)
            {
                loss -= logProb;
                // grad = prob - 1
                gradOut[idx * numClasses + c] = prob - 1.0f;
            }
            else
            {
                gradOut[idx * numClasses + c] = prob;
            }
        }

        atomicAdd(lossOut, loss);
    }
}

float trainBatch(Network &net, Optimizer &opt, float *h_inputBatch, float *h_labelBatch,
                 int batchSize, int inputSize, int numClasses)
{
    // Create input tensor on device
    Tensor inputTensor(batchSize * inputSize);
    inputTensor.copyFromHost(h_inputBatch, batchSize * inputSize);

    // Forward
    Tensor *output = net.forward(&inputTensor); // shape: [batchSize, numClasses]

    // Prepare gradient wrt output
    Tensor gradOutput(batchSize * numClasses);

    // For calculating the loss, we do a device-side kernel
    Tensor d_loss(1); // single float to accumulate the batch's loss
    cudaMemset(d_loss.getData(), 0, sizeof(float));

    // Copy labels to device
    Tensor d_labels(batchSize);
    d_labels.copyFromHost(h_labelBatch, batchSize);

    dim3 block(256);
    dim3 grid((batchSize + block.x - 1) / block.x);
    computeLossAndGrad<<<grid, block>>>(
        output->getData(), d_labels.getData(),
        gradOutput.getData(), d_loss.getData(),
        batchSize, numClasses);

    // Backward
    Tensor *gradInput = net.backward(&gradOutput);

    // Update parameters
    auto paramsAndGrads = net.getAllParamsAndGrads();
    opt.step(paramsAndGrads);

    // Copy loss back to host
    float hostLoss;
    d_loss.copyToHost(&hostLoss, 1);

    // Return average loss
    return hostLoss / batchSize;
}

float evaluate(Network &net, float *h_input, float *h_labels, int dataSize, int inputSize, int numClasses, int batchSize)
{
    // We'll compute accuracy
    int correct = 0;

    for (int start = 0; start < dataSize; start += batchSize)
    {
        int end = std::min(start + batchSize, dataSize);
        int currentBatch = (end - start);

        Tensor inputTensor(currentBatch * inputSize);

        inputTensor.copyFromHost(&h_input[start * inputSize], currentBatch * inputSize);

        Tensor *output = net.forward(&inputTensor);

        // Copy the output back to host
        std::vector<float> hostOut(currentBatch * numClasses);
        output->copyToHost(hostOut.data(), currentBatch * numClasses);

        // Compare predicted label with ground-truth label
        for (int i = 0; i < currentBatch; i++)
        {
            int label = (int)h_labels[start + i];
            float maxVal = -1e9;
            int maxIdx = -1;
            for (int c = 0; c < numClasses; c++)
            {
                float val = hostOut[i * numClasses + c];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = c;
                }
            }
            if (maxIdx == label)
            {
                correct++;
            }
        }
    }
    return 100.f * (float)correct / (float)dataSize;
}

int main()
{
    // Load MNIST dataset
    MNISTDataset dataset(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
        "data/t10k-images.idx3-ubyte",
        "data/t10k-labels.idx1-ubyte");

    std::cout << "Loaded MNIST dataset" << std::endl;

    int inputSize = 28 * 28; // MNIST
    int numClasses = 10;
    int batchSize = 256;
    int epochs = 20;
    float learningRate = 1e-4;

    // Build network
    Network net;
    net.addLayer(new LinearLayer(inputSize, 256, batchSize));
    net.addLayer(new ReLULayer());
    net.addLayer(new LinearLayer(256, numClasses, batchSize));

    // Create optimizer
    Optimizer opt(learningRate);

    // Training loop
    int trainSize = dataset.trainSize;
    std::cout << "Training size: " << trainSize << std::endl;

    for (int e = 0; e < epochs; e++)
    {
        auto start = std::chrono::high_resolution_clock::now();  // Start timing
        std::cout << "Epoch " << (e + 1) << " / " << epochs << std::endl;

        float epochLoss = 0.0f;
        int batches = 0;

        for (int i = 0; i < trainSize; i += batchSize)
        {
            // std::cout << "i = " << i << "\n";
            int currentBatch = std::min(batchSize, trainSize - i);

            // Prepare the batch on host
            std::vector<float> inputBatch(currentBatch * inputSize);
            std::vector<float> labelBatch(currentBatch);

            for (int b = 0; b < currentBatch; b++)
            {
                for (int p = 0; p < inputSize; p++)
                {
                    inputBatch[b * inputSize + p] = dataset.trainImages[(i + b) * inputSize + p];
                }
                labelBatch[b] = dataset.trainLabels[i + b];
            }
            // std::cout << "Start training batch " << i / batchSize << std::endl;
            float lossVal = trainBatch(net, opt, inputBatch.data(), labelBatch.data(),
                                       currentBatch, inputSize, numClasses);

            epochLoss += lossVal;
            batches++;
        }

        std::cout << "  Avg loss: " << (epochLoss / batches) << std::endl;
        // Evaluate on test set
        float testAcc = evaluate(net, dataset.testImages.data(), dataset.testLabels.data(),
                                dataset.testSize, inputSize, numClasses, batchSize);
        std::cout << "Test Accuracy: " << testAcc << "%" << std::endl;
        auto end = std::chrono::high_resolution_clock::now();  // End timing
        std::chrono::duration<float> duration = end - start;
        std::cout << "  Time: " << duration.count() << " seconds" << std::endl;
    }
    

    return 0;
}

// __global__ void helloFromGPU() {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     printf("Hello from GPU thread %d\n", idx);
// }

// int main() {
//     // Launch kernel with 1 block and 5 threads
//     helloFromGPU<<<256, 64>>>();

//     // Wait for GPU to finish before accessing the output
//     cudaDeviceSynchronize();

//     return 0;
// }