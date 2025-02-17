#include "cuda_runtime.h"
#include "kernel.h"
#include "stdio.h"

__device__ void insertAtIndex(const int sourcePtr, const int length, const int destinationPtr, const int indexToInsert, const int value, char *mem)
{
    int sourceIndex = 0;
    int destinationIndex = 0;

    while (destinationIndex < length + 1)
    {
        if (destinationIndex != indexToInsert)
        {
            mem[destinationPtr + destinationIndex] = mem[sourcePtr + sourceIndex];
            sourceIndex++;
            destinationIndex++;
        }
        else
        {
            mem[destinationPtr + destinationIndex] = value;
            destinationIndex++;
        }
    }
}

__device__ void removeAtIndex(const int sourcePtr, const int length, const int destinationPtr, const int indexToRemove, char *mem)
{
    int sourceIndex = 0;
    int destinationIndex = 0;

    while (destinationIndex < length - 1)
    {
        if (sourceIndex != indexToRemove)
        {
            mem[destinationPtr + destinationIndex] = mem[sourcePtr + sourceIndex];
            sourceIndex++;
            destinationIndex++;
        }
        else
        {
            sourceIndex++;
        }
    }
}

__device__ void getDimensionalIndices(int flatIndex, const int shapePtr, const int shapeSize, const int destinationPtr, char *mem)
{
    int currentIndex = flatIndex;

    for (int i = 0; i < shapeSize; i++)
    {
        mem[destinationPtr + i] = currentIndex % mem[shapePtr + i];
        currentIndex = currentIndex / mem[shapePtr + i];
    }
}

__device__ int getFlatIndex(const int dimensionalIndexPtr, const int shapePtr, const int shapeSize, char *mem)
{
    int index = 0;
    for (int i = 0; i < shapeSize; i++)
    {
        index *= mem[shapePtr + shapeSize - 1 - i];
        index += mem[dimensionalIndexPtr + shapeSize - 1 - i];
    }
    return index;
}

__global__ void addKernel(const int size, const float *valuesA, const float *valuesB, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = valuesA[idx] + valuesB[idx];
    }
}
__global__ void addBackpropKernel(const int size, float *gradA, float *gradB, const float *gradResult)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        gradA[idx] += gradResult[idx];
        gradB[idx] += gradResult[idx];
    }
}
__global__ void subKernel(const int size, const float *valuesA, const float *valuesB, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = valuesA[idx] - valuesB[idx];
    }
}
__global__ void subBackpropKernel(const int size, float *gradA, float *gradB, const float *gradResult)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        gradA[idx] += gradResult[idx];
        gradB[idx] += -1 * gradResult[idx];
    }
}
__global__ void mulKernel(const int size, const float *valuesA, const float *valuesB, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = valuesA[idx] * valuesB[idx];
    }
}
__global__ void mulBackpropKernel(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        gradA[idx] += valuesB[idx] * gradResult[idx];
        gradB[idx] += valuesA[idx] * gradResult[idx];
    }
}
__global__ void divKernel(const int size, const float *valuesA, const float *valuesB, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = valuesA[idx] / valuesB[idx];
    }
}
__global__ void divBackpropKernel(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        gradA[idx] += 1 / valuesB[idx] * gradResult[idx];
        gradB[idx] += -1 * valuesA[idx] / pow(valuesB[idx], 2) * gradResult[idx];
    }
}
__global__ void powKernel(const int size, const float *valuesA, const float *valuesB, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = pow(valuesA[idx], valuesB[idx]);
    }
}
__global__ void powBackpropKernel(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        gradA[idx] += valuesB[idx] * pow(valuesA[idx], valuesB[idx] - 1) * gradResult[idx];
        gradB[idx] += log(valuesA[idx]) * pow(valuesA[idx], valuesB[idx]) * gradResult[idx];
    }
}
__global__ void negKernel(const int size, const float *values, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = -values[idx];
    }
}
__global__ void negBackpropKernel(const int size, float *grad, const float *gradResult)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        grad[idx] += -1 * gradResult[idx];
    }
}
__global__ void expKernel(const int size, const float *values, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = exp(values[idx]);
    }
}
__global__ void expBackpropKernel(const int size, float *grad, const float *gradResult, const float *values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        grad[idx] += exp(values[idx]) * gradResult[idx];
    }
}
__global__ void tanhKernel(const int size, const float *values, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = tanh(values[idx]);
    }
}
__global__ void tanhBackpropKernel(const int size, float *grad, const float *gradResult, const float *values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        grad[idx] += (1 - pow(tanh(values[idx]), 2)) * gradResult[idx];
    }
}
__global__ void sumKernel(const int *shape, const int shapeSize, const int dimToReduce, const float *values, float *output, char *mem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int newSize = 1;
    for (int j = 0; j < shapeSize; j++)
    {
        if (j != dimToReduce)
        {
            newSize *= shape[j];
        }
    }
    if (idx < newSize)
    {
        size_t basePtr = idx * ((shapeSize * 4) - 2); //manual calc :/
        size_t memPtr = basePtr;
        size_t shapePtr = basePtr;

        for (int i = 0; i < shapeSize; i++)
        {
            mem[memPtr] = shape[i];
            memPtr++;
        }

        size_t newShapePtr = memPtr;
        removeAtIndex(shapePtr, shapeSize, newShapePtr, dimToReduce, mem);
        memPtr += shapeSize - 1;

        size_t partialDimIndexPtr = memPtr;
        getDimensionalIndices(idx, newShapePtr, shapeSize - 1, partialDimIndexPtr, mem);
        memPtr += shapeSize - 1;

        for (int i = 0; i < shape[dimToReduce]; i++)
        {
            size_t dimIndexPtr = memPtr;
            insertAtIndex(partialDimIndexPtr, shapeSize - 1, dimIndexPtr, dimToReduce, i, mem);

            size_t flatIdx = getFlatIndex(dimIndexPtr, shapePtr, shapeSize, mem);
            output[idx] += values[flatIdx];
        }
    }
}

__global__ void sumBackpropKernel(const int *shape, const int shapeSize, const int dimToReduce, float *grad, const float *gradResult, char* mem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int inSize = 1;
    for (int j = 0; j < shapeSize; j++)
    {
        inSize *= shape[j];
    }

    if (idx < inSize)
    {
        size_t basePtr = idx * ((shapeSize * 4) - 2); //manual calc :/
        size_t memPtr = basePtr;
        size_t shapePtr = basePtr;

        for(int i = 0; i < shapeSize; i++){
            mem[memPtr] = shape[i];
            memPtr++;
        }

        size_t outShapePtr = memPtr;
        removeAtIndex(shapePtr, shapeSize, outShapePtr, dimToReduce, mem);
        memPtr += shapeSize - 1;

        size_t inDimIndexPtr = memPtr;
        getDimensionalIndices(idx, shapePtr, shapeSize, inDimIndexPtr, mem);
        memPtr += shapeSize;

        size_t outDimIndexPtr = memPtr;
        removeAtIndex(inDimIndexPtr, shapeSize, outDimIndexPtr, dimToReduce, mem);
        memPtr += shapeSize - 1;


        size_t outputFlatIdx = getFlatIndex(outDimIndexPtr, outShapePtr, shapeSize - 1, mem);

        grad[idx] += gradResult[outputFlatIdx];
    }
}

// Binary Ops

float *add_op(const int size, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *valuesA_d;
    float *valuesB_d;
    float *output_d;

    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, valuesA_d, valuesB_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
    cudaFree(output_d);

    return output;
}

void addBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *gradA_d;
    float *gradB_d;
    float *gradResult_d;

    cudaMalloc((void **)&gradA_d, size * sizeof(float));
    cudaMalloc((void **)&gradB_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));

    cudaMemcpy(gradA_d, gradA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradB_d, gradB, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);

    addBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradResult_d);

    if (gradA == gradB)
    { // if same reference then we need to add them
        float *gradUnified_d;
        cudaMalloc((void **)&gradUnified_d, size * sizeof(float));
        addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradUnified_d);

        cudaMemcpy(gradA, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(gradUnified_d);
    }
    else
    {
        cudaMemcpy(gradA, gradA_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradB_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(gradA_d);
    cudaFree(gradB_d);
    cudaFree(gradResult_d);
}

float *sub_op(const int size, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *valuesA_d;
    float *valuesB_d;
    float *output_d;

    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    subKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, valuesA_d, valuesB_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
    cudaFree(output_d);

    return output;
}

void subBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *gradA_d;
    float *gradB_d;
    float *gradResult_d;

    cudaMalloc((void **)&gradA_d, size * sizeof(float));
    cudaMalloc((void **)&gradB_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));

    cudaMemcpy(gradA_d, gradA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradB_d, gradB, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);

    subBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradResult_d);

    if (gradA == gradB)
    { // if same reference then we need to add them
        float *gradUnified_d;
        cudaMalloc((void **)&gradUnified_d, size * sizeof(float));
        addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradUnified_d);

        cudaMemcpy(gradA, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(gradUnified_d);
    }
    else
    {
        cudaMemcpy(gradA, gradA_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradB_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(gradA_d);
    cudaFree(gradB_d);
    cudaFree(gradResult_d);
}

float *mul_op(const int size, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *valuesA_d;
    float *valuesB_d;
    float *output_d;

    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    mulKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, valuesA_d, valuesB_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
    cudaFree(output_d);

    return output;
}

void mulBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *gradA_d;
    float *gradB_d;
    float *gradResult_d;
    float *valuesA_d;
    float *valuesB_d;

    cudaMalloc((void **)&gradA_d, size * sizeof(float));
    cudaMalloc((void **)&gradB_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));
    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));

    cudaMemcpy(gradA_d, gradA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradB_d, gradB, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    mulBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradResult_d, valuesA_d, valuesB_d);

    if (gradA == gradB)
    { // if same reference then we need to add them
        float *gradUnified_d;
        cudaMalloc((void **)&gradUnified_d, size * sizeof(float));
        addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradUnified_d);

        cudaMemcpy(gradA, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(gradUnified_d);
    }
    else
    {
        cudaMemcpy(gradA, gradA_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradB_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(gradA_d);
    cudaFree(gradB_d);
    cudaFree(gradResult_d);
    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
}

float *div_op(const int size, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *valuesA_d;
    float *valuesB_d;
    float *output_d;

    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    divKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, valuesA_d, valuesB_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
    cudaFree(output_d);

    return output;
}

void divBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *gradA_d;
    float *gradB_d;
    float *gradResult_d;
    float *valuesA_d;
    float *valuesB_d;

    cudaMalloc((void **)&gradA_d, size * sizeof(float));
    cudaMalloc((void **)&gradB_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));
    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));

    cudaMemcpy(gradA_d, gradA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradB_d, gradB, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    divBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradResult_d, valuesA_d, valuesB_d);

    if (gradA == gradB)
    { // if same reference then we need to add them

        float *ggA = (float *)malloc(size * sizeof(float));
        float *ggB = (float *)malloc(size * sizeof(float));
        cudaMemcpy(ggA, gradA_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ggB, gradB_d, size * sizeof(float), cudaMemcpyDeviceToHost);

        float *gradUnified_d;
        cudaMalloc((void **)&gradUnified_d, size * sizeof(float));
        addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradUnified_d);

        cudaMemcpy(gradA, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(gradUnified_d);
    }
    else
    {
        cudaMemcpy(gradA, gradA_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradB_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(gradA_d);
    cudaFree(gradB_d);
    cudaFree(gradResult_d);
    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
}

float *pow_op(const int size, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *valuesA_d;
    float *valuesB_d;
    float *output_d;

    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    powKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, valuesA_d, valuesB_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
    cudaFree(output_d);

    return output;
}

void powBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *gradA_d;
    float *gradB_d;
    float *gradResult_d;
    float *valuesA_d;
    float *valuesB_d;

    cudaMalloc((void **)&gradA_d, size * sizeof(float));
    cudaMalloc((void **)&gradB_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));
    cudaMalloc((void **)&valuesA_d, size * sizeof(float));
    cudaMalloc((void **)&valuesB_d, size * sizeof(float));

    cudaMemcpy(gradA_d, gradA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradB_d, gradB, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesA_d, valuesA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesB_d, valuesB, size * sizeof(float), cudaMemcpyHostToDevice);

    powBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradResult_d, valuesA_d, valuesB_d);

    if (gradA == gradB)
    { // if same reference then we need to add them
        float *gradUnified_d;
        cudaMalloc((void **)&gradUnified_d, size * sizeof(float));
        addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, gradA_d, gradB_d, gradUnified_d);

        cudaMemcpy(gradA, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradUnified_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(gradUnified_d);
    }
    else
    {
        cudaMemcpy(gradA, gradA_d, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradB, gradB_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(gradA_d);
    cudaFree(gradB_d);
    cudaFree(gradResult_d);
    cudaFree(valuesA_d);
    cudaFree(valuesB_d);
}

// Unary Ops
float *neg_op(const int size, const float *values)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *values_d;
    float *output_d;

    cudaMalloc((void **)&values_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(values_d, values, size * sizeof(float), cudaMemcpyHostToDevice);

    negKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, values_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(values_d);
    cudaFree(output_d);

    return output;
}
void negBackprop_op(const int size, float *grad, const float *gradResult)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *grad_d;
    float *gradResult_d;

    cudaMalloc((void **)&grad_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));

    cudaMemcpy(grad_d, grad, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);

    negBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, grad_d, gradResult_d);

    cudaMemcpy(grad, grad_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(grad_d);
    cudaFree(gradResult_d);
}
float *exp_op(const int size, const float *values)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *values_d;
    float *output_d;

    cudaMalloc((void **)&values_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(values_d, values, size * sizeof(float), cudaMemcpyHostToDevice);

    expKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, values_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(values_d);
    cudaFree(output_d);

    return output;
}
void expBackprop_op(const int size, float *grad, const float *gradResult, const float *values)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *grad_d;
    float *gradResult_d;
    float *values_d;

    cudaMalloc((void **)&grad_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));
    cudaMalloc((void **)&values_d, size * sizeof(float));

    cudaMemcpy(grad_d, grad, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values, size * sizeof(float), cudaMemcpyHostToDevice);

    expBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, grad_d, gradResult_d, values_d);

    cudaMemcpy(grad, grad_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(grad_d);
    cudaFree(gradResult_d);
    cudaFree(values_d);
}
float *tanh_op(const int size, const float *values)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[size];

    float *values_d;
    float *output_d;

    cudaMalloc((void **)&values_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, size * sizeof(float));

    cudaMemcpy(values_d, values, size * sizeof(float), cudaMemcpyHostToDevice);

    tanhKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, values_d, output_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(values_d);
    cudaFree(output_d);

    return output;
}
void tanhBackprop_op(const int size, float *grad, const float *gradResult, const float *values)
{
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *grad_d;
    float *gradResult_d;
    float *values_d;

    cudaMalloc((void **)&grad_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));
    cudaMalloc((void **)&values_d, size * sizeof(float));

    cudaMemcpy(grad_d, grad, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values, size * sizeof(float), cudaMemcpyHostToDevice);

    tanhBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(size, grad_d, gradResult_d, values_d);

    cudaMemcpy(grad, grad_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(grad_d);
    cudaFree(gradResult_d);
    cudaFree(values_d);
}

// Reduction Ops

float *sum_op(const int *shape, const int shapeSize, const int dimToReduce, const float *values)
{
    int BLOCK_SIZE = 32;

    int size = 1;
    for (int i = 0; i < shapeSize; i++)
    {
        size *= shape[i];
    }
    int outSize = size / shape[dimToReduce];

    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *output = new float[outSize];

    float *values_d;
    float *output_d;
    int *shape_d;
    char*mem_d;

    cudaMalloc((void **)&values_d, size * sizeof(float));
    cudaMalloc((void **)&output_d, outSize * sizeof(float));
    cudaMalloc((void **)&shape_d, shapeSize * sizeof(int));

    // memory areana
    int threadMemSize = ((shapeSize * 4) - 2) * 4;
    cudaMalloc((void **)&mem_d, threadMemSize * outSize);

    cudaMemcpy(values_d, values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_d, shape, shapeSize * sizeof(int), cudaMemcpyHostToDevice);

    sumKernel<<<GRID_SIZE, BLOCK_SIZE>>>(shape_d, shapeSize, dimToReduce, values_d, output_d, mem_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, output_d, outSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(values_d);
    cudaFree(output_d);
    cudaFree(shape_d);
    cudaFree(mem_d);

    return output;
}

void sumBackprop_op(const int *shape, const int shapeSize, const int dimToReduce, float *grad, const float *gradResult)
{
    int BLOCK_SIZE = 32;

    int size = 1;
    for (int i = 0; i < shapeSize; i++)
    {
        size *= shape[i];
    }
    int outSize = size / shape[dimToReduce];

    int GRID_SIZE = (int)ceil(size / (float)BLOCK_SIZE);

    float *grad_d;
    float *gradResult_d;
    int *shape_d;
    char *mem_d;

    cudaMalloc((void **)&grad_d, size * sizeof(float));
    cudaMalloc((void **)&gradResult_d, size * sizeof(float));
    cudaMalloc((void **)&shape_d, shapeSize * sizeof(int));

    // memory areana
    int threadMemSize = ((shapeSize * 4) - 2) * 4;
    cudaMalloc((void **)&mem_d, threadMemSize * outSize);

    cudaMemcpy(grad_d, grad, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gradResult_d, gradResult, outSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_d, shape, shapeSize * sizeof(int), cudaMemcpyHostToDevice);

    sumBackpropKernel<<<GRID_SIZE, BLOCK_SIZE>>>(shape_d, shapeSize, dimToReduce, grad_d, gradResult_d, mem_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(grad, grad_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(grad_d);
    cudaFree(gradResult_d);
    cudaFree(shape_d);
    cudaFree(mem_d);
}