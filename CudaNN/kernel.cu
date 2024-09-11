
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Tensor2D.cuh"
#include <iostream>
#include <math.h>

 //function to add the elements of two arrays
__global__ void add(int n, float* x, float* y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    while (index < n) {
        y[index] = x[index] + y[index];
        index += stride;
    }
}

int main(void)
{
    printf("beg\n");
    auto cpu_tensor1 = new float [550][370];
    auto cpu_tensor2 = new float [370][260];
    auto cpu_tensor3 = new float [550][260];

    printf("Init\n");
    Tensor2D tensor1 = Tensor2D::Tensor2D(550, 370);
    for (int i = 0; i < tensor1.columns; i++) {
        for (int j = 0; j < tensor1.rows; j++) {
            tensor1.set_value(j, i, j * tensor1.columns + i);
            cpu_tensor1[j][i] = j * tensor1.columns + i;
        }
    }

    Tensor2D tensor2 = Tensor2D::Tensor2D(370, 260); 
    for (int i = 0; i < tensor2.columns; i++) {
        for (int j = 0; j < tensor2.rows; j++) {
            tensor2.set_value(j, i, j * tensor2.columns + tensor2.columns - i);
            cpu_tensor2[j][i] = j * tensor2.columns + tensor2.columns - i;
        }
    }

    Tensor2D tensor3 = Tensor2D::Tensor2D(550, 260);
    tensor1.tensor_multiply(tensor3, tensor2);

    for (int j = 0; j < tensor3.rows; j++) {
        for (int i = 0; i < tensor3.columns; i++) {
            float val = 0;
            for (int k = 0; k < tensor1.columns; k++) {
                val += cpu_tensor1[j][k] * cpu_tensor2[k][i];
            }
            if (val != tensor3.get_value(j, i)) {
                printf("Issue with row: %i col: %i", j, i);
            }
        }
    }
    printf("Done \n");
    free(cpu_tensor1);
    free(cpu_tensor2);
    free(cpu_tensor3);
    //printf("Tensor1: \n");
    //tensor1.print_data();
    //printf("Tensor2: \n");
    //tensor2.print_data();
    //printf("Tensor3: \n");
    //tensor3.print_data();
    //tensor.multiply(6.45);
    //tensor.print_data();
    //int N = 1 << 20; // 1M elements

    //float* x;
    //float* y;

    //cudaMallocManaged(&x, N*sizeof(float));
    //cudaMallocManaged(&y, N*sizeof(float));

    //// initialize x and y arrays on the host
    //for (int i = 0; i < N; i++) {
    //    x[i] = 1.0f;
    //    y[i] = 2.0f;
    //}

    //// Run kernel on 1M elements on the CPU
    //add<<<1, 256>>>(N, x, y);
    //cudaDeviceSynchronize();
    //// Check for errors (all values should be 3.0f)
    //float maxError = 0.0f;
    //for (int i = 0; i < N; i++)
    //    maxError = fmax(maxError, fabs(y[i] - 3.0f));
    //std::cout << "Max error: " << maxError << std::endl;

    ////// Free memory
    //cudaFree(x);
    //cudaFree(y);

    return 0;
}
