#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <cstring> // For memset if needed
#include <iostream> // Optional: for debugging

#define assertCudaError(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

class Tensor {
private:
    int size;
    float* d_data;

public:
    // Constructor: allocates and zero-initializes device memory
    Tensor(int size_) : size(size_), d_data(nullptr) {
        assert(size > 0);
        cudaError_t err;

        err = cudaMalloc(&d_data, size * sizeof(float));
        // assert(err == cudaSuccess);
        assertCudaError(err);

        err = cudaMemset(d_data, 0, size * sizeof(float));
        assertCudaError(err);

        // std::cerr << "Successfully allocated device memory of size: " << size * sizeof(float) << " bytes" << std::endl;
    }

    // Destructor: frees device memory
    ~Tensor() {
        if (d_data) {
            cudaFree(d_data);
            // std::cerr << "Successfully freed device memory of size: " << size * sizeof(float) << " bytes" << std::endl;
        }
    }

    // Returns the number of elements
    int getSize() const {
        return size;
    }

    // Returns a pointer to device memory (non-const)
    float* getData() {
        return d_data;
    }

    // Returns a pointer to device memory (const)
    const float* getData() const {
        return d_data;
    }

    // Copies data from host to device
    void copyFromHost(const float* hostPtr, int numElements) {
        assert(numElements <= size);
        cudaError_t err = cudaMemcpy(d_data, hostPtr, numElements * sizeof(float), cudaMemcpyHostToDevice);
        assertCudaError(err);
    }

    // Copies data from device to host
    void copyToHost(float* hostPtr, int numElements) const {
        assert(numElements <= size);
        cudaError_t err = cudaMemcpy(hostPtr, d_data, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        assertCudaError(err);
    }
};

#endif // TENSOR_CUH
