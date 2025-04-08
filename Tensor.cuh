#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

class Tensor {
private:
    float* d_data;
    int size;  // total number of floats

public:
    Tensor(int size)
        : d_data(nullptr), size(size)
    {
        allocate(size);
    }

    ~Tensor() {
        freeMemory();
    }

    int getSize() const { return size; }

    float* getData() { return d_data; }
    const float* getData() const { return d_data; }

    // Copy from host to device
    void copyFromHost(const float* hostPtr, int numElements) {
        assert(numElements <= size);
        cudaMemcpy(d_data, hostPtr, numElements * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Copy from device to host
    void copyToHost(float* hostPtr, int numElements) const {
        if (numElements > size) {
            std::cout << "WTF: numElements = " << numElements << " size = " << size << std::endl;
        }
        assert(numElements <= size);
        cudaMemcpy(hostPtr, d_data, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    }

private:
    void allocate(int s) {
        cudaMalloc(&d_data, s * sizeof(float));
        cudaMemset(d_data, 0, s * sizeof(float));  // zero initialization
    }

    void freeMemory() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }
};

#endif // TENSOR_CUH