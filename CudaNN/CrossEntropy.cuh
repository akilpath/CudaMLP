#ifndef CROSS_ENTROPY_CUH
#define CROSS_ENTROPY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Tensor2D.cuh"


class CrossEntropy {
public:
	CrossEntropy();
	~CrossEntropy();
	float calculate_loss(Tensor2D& target, Tensor2D& prediction);
	int calculate_gradients(Tensor2D& target, Tensor2D& prediction, Tensor2D& out_err);


};

#endif // !CROSS_ENTROPY_CUH