#include "ReLu.cuh"

__global__ void _relu_(float* out, float* data, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < size) {
		out[index] = fmaxf(0, data[index]);
		index += stride;
	}
}

__global__ void _relu_grad(float* out, float* in_data, float* gradients, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		if (in_data[index] > 0) {
			out[index] = gradients[index];
		}
		else {
			out[index] = 0;
		}
	}
}

ReLu::ReLu() {
	input_data_ = nullptr;
	output_data_ = nullptr;
}

ReLu::~ReLu() {
	if (input_data_ != nullptr) delete input_data_;
	if (output_data_ != nullptr) delete output_data_;
}

int ReLu::forward(Tensor2D& out, Tensor2D& in)
{
	if (out.rows() != in.rows() || out.columns() != in.columns())
	{
		std::cerr << "ReLu::forward: input and output dimensions do not match" << std::endl;
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

	input_data_ = new Tensor2D(in.rows(), in.columns());
	output_data_ = new Tensor2D(out.rows(), out.columns());
	in.copy(*input_data_);
	int size = in.rows() * in.columns();
	int block_size = 128;
	int num_blocks = ceil(((float) size) / block_size);
	if (num_blocks == 0) num_blocks = 1;

	_relu_<<<num_blocks, block_size>>>(out.data_, in.data_, size);
	cudaDeviceSynchronize();
	out.copy(*output_data_);
	return 0;
}

int ReLu::backward(Tensor2D& out_err, Tensor2D& in_err)
{
	if (out_err.rows() != in_err.rows() || out_err.columns() != in_err.columns())
	{
		printf("ReLu::backward: input and output dimensions do not match");
		return -1;
	}
	if (input_data_ == nullptr)
	{
		printf("ReLu::backward: input data is null");
		return -1;
	}
	if (input_data_->rows() != out_err.rows() || input_data_->columns() != out_err.columns())
	{
		printf("ReLu::backward: input data dimensions do not match output error dimensions");
		return -1;
	}

	int block_size = 128;
	int num_blocks = ceil(((float)in_err.total_elements()) / block_size);
	if (num_blocks == 0) num_blocks = 1;

	//printf("in_err: \n");
	//in_err.print_data();
	//printf("input_data_: \n");
	//input_data_->print_data();


	_relu_grad << <num_blocks, block_size >> > (out_err.data_, input_data_ -> data_, in_err.data_, in_err.total_elements());
	cudaDeviceSynchronize();
	//printf("out_err: \n");
	//out_err.print_data();

	if (input_data_ != nullptr) {
		delete input_data_;
		input_data_ = nullptr;
	}
	if (output_data_ != nullptr) {
		delete output_data_;
		output_data_ = nullptr;
	}
	return 0;
}