#include "ReLu.cuh"

__global__ void _relu_(float* out, float* data, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < size) {
		out[index] = fmaxf(0, data[index]);
		index += stride;
	}
}

int ReLu::forward(Tensor2D& out, Tensor2D& in)
{
	Tensor2D* input_data_ = new Tensor2D(in.rows(), in.columns());
	Tensor2D* output_data_ = new Tensor2D(out.rows(), out.columns());
	in.copy(*input_data_);
	int size = in.rows() * in.columns();


	return 0;
}

int ReLu::backward(Tensor2D& out_err, Tensor2D& in_err)
{
	return 0;
}