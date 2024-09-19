#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "Tensor2D.cuh"
#include "VectorND.cuh"

class SoftMax {
	public:
		SoftMax();
		~SoftMax();
		int forward(Tensor2D& out, Tensor2D& in);
		int backward(Tensor2D& out_err, Tensor2D& in_err);
	private:
		Tensor2D* input_data_;
		Tensor2D* output_data_;
};

#endif

