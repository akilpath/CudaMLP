
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Tensor2D.cuh"
#include "VectorND.cuh"
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
    Tensor2D tensor1 = Tensor2D::Tensor2D(5,3);
    for (int i = 0; i < tensor1.columns(); i++) {
        for (int j = 0; j < tensor1.rows(); j++) {
            tensor1.set_value(j, i, j * tensor1.columns() + i);
        }
    }

    Tensor2D tensor2 = Tensor2D::Tensor2D(3, 4); 
    for (int i = 0; i < tensor2.columns(); i++) {
        for (int j = 0; j < tensor2.rows(); j++) {
            tensor2.set_value(j, i, j * tensor2.columns() + tensor2.columns() - i);
        }
    }

    Tensor2D tensor3 = Tensor2D::Tensor2D(5, 4);
    int result = tensor1.tensor_multiply(tensor3, tensor2);

	if (result != 0) {
		std::cout << "Error in tensor multiply" << std::endl;
	}

    tensor3.print_data();

	float arr[5] = { 1, 2, 3, 4, 5};
	VectorND vec1 = VectorND::VectorND(5, arr);

	Tensor2D tensor4 = Tensor2D::Tensor2D(5, 4);

	int res2 = tensor3.add_vector_to_columns(tensor4, vec1);
	if (res2 != 0) {
		std::cout << "Error in add vector to columns" << std::endl;
	}
    std::cout << "Tensor4" << std::endl;
	tensor4.print_data();

	Tensor2D tensor5 = Tensor2D::Tensor2D(5, 4);
    tensor4.copy(tensor5);

	tensor5.tensor_add(tensor5, tensor4);
    std::cout << "Tensor 5" << std::endl;
	tensor5.print_data();

	float arr2[5] = { 9, 4, 3, 2, 1 };
	VectorND vec2 = VectorND(5, arr2);
	
	float arr3[5] = { 5, 2, 3, 3, 5 };
	VectorND vec3 = VectorND(5, arr3);
	VectorND vec4 = VectorND(5);

    std::cout << "Vector 2" << std::endl;
	vec2.print();
	std::cout << "Vector 3" << std::endl;
	vec3.print();
	int result_vec = vec3.vector_add(vec3, vec2);
	std::cout << "Result_vec " << result_vec<<" Vector 3 after adding vector 2" << std::endl;
    vec3.print();
    

    return 0;
}
