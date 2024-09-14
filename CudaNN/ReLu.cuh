#ifndef RELU_CUH
#define RELU_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "Tensor2D.cuh"
#include "VectorND.cuh"

class ReLu {
	public:
		ReLu();
		~ReLu();
		int forward(Tensor2D& out, Tensor2D& in);
		int backward(Tensor2D& out_err, Tensor2D& in_err);
		int forward(VectorND& out, VectorND& in);
		int backward(VectorND& out, VectorND& in);
	private:
		Tensor2D* input_data_;
		Tensor2D* output_data_;
};

#endif
