#include "SoftMax.cuh"

__global__ void _softmax(float* out, float* in, int rows, int columns) {
	//currently done per column, need to implement shared memory for matrix wise
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < columns) {
		float sum = 0;
		float max = -10000;
		for (int i = 0; i < rows; i++) {
			if (in[i * columns + index] > max) {
				max = in[i * columns + index];
			}
		}

		for (int i = 0; i < rows; i++) {
			//if (index == 4) {
			//	printf("index: %d, sum: %f, exp: %f\n", index, sum, exp(in[i * columns + index] - max));
			//}
			sum += exp(in[i * columns + index] - max);
		}

		for (int i = 0; i < rows; i++) {
			
			out[i * columns + index] = exp(in[i * columns + index] - max) / sum;
		}
	}
}

SoftMax::SoftMax() {
	input_data_ = new Tensor2D();
	output_data_ = new Tensor2D();
}

SoftMax::~SoftMax() {
	delete input_data_;
	delete output_data_;
}

int SoftMax::forward(Tensor2D&out, Tensor2D&in) {
	if (out.rows() != in.rows() || out.columns() != in.columns())
	{
		printf("Error in SoftMax::forward, outputand input dimensions do not match\n");
		return -1;
	}
	if (input_data_ != nullptr)
	{
		delete input_data_;
	}
	if (output_data_ != nullptr)
	{
		delete output_data_;
	}

	int block_size = 128;
	int num_blocks = ceil(((float)in.columns()) / block_size);
	if (num_blocks == 0) num_blocks = 1;
	input_data_ = new Tensor2D(in.rows(), in.columns());
	output_data_ = new Tensor2D(out.rows(), out.columns());
	in.copy(*input_data_);
	_softmax << <num_blocks, block_size >> > (out.data_, in.data_, in.rows(), in.columns());
	cudaDeviceSynchronize();
	out.copy(*output_data_);

	return 0;
}

int SoftMax::backward(Tensor2D& out_err, Tensor2D& in_err) {
	return 0;
}