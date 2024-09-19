
#ifndef MSE_CUH
#define MSE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"
#include <iostream>
#include "Tensor2D.cuh"
#include "VectorND.cuh"

class MSE {
public:
	MSE();
	~MSE();
	float calculate_loss(Tensor2D& target, Tensor2D& prediction);
	int calculate_gradients(Tensor2D& target, Tensor2D& prediction, Tensor2D& out_err);

};

#endif