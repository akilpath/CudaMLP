#include "VectorND.cuh"



__global__ void _add_vector(float* out, float* data_1, float* data_2, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < size) {
		out[index] = data_1[index] + data_2[index];
		index += stride;
	}
}

__global__ void _subtract_vector(float* out, float* data_1, float* data_2, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < size) {
		out[index] = data_1[index] - data_2[index];
		index += stride;
	}
}

__global__ void _multiply_vector(float* out, float* data_1, float* data_2, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < size) {
		out[index] = data_1[index] * data_2[index];
		index += stride;
	}
}

__global__ void _scalar_multiply(float* data, float scalar, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < size) {
		data[index] = data[index] * scalar;
		index += stride;
	}
}

VectorND::VectorND(int size) {
	this->size_ = size;
	cudaMallocManaged(&data_, size * sizeof(float));
	for (int i = 0; i < size; i++) {
		this->data_[i] = 0;
	}
}

VectorND::VectorND(int size, float* in_data)
{
	this->size_ = size;
	cudaMallocManaged(&data_, size * sizeof(float));
	for (int i = 0; i < size; i++) {
		this->data_[i] = in_data[i];
	}
}
VectorND::VectorND() {
	this->size_ = 0;
	this->data_ = nullptr;
}

VectorND::~VectorND() {
	if (data_ != nullptr) {
		cudaFree(data_);
	}
}

int VectorND::size(){
	return this->size_;
}


void VectorND::print() const {
	std::cout << "[";
	for (int i = 0; i < this->size_; i++) {
		if (i == this->size_ - 1)
			std::cout << this->data_[i];
		else
			std::cout << this->data_[i] << ", ";
	}
	std::cout << "]" << std::endl;
}

int VectorND::scalar_multiply(VectorND &out, float scalar)
{
	if (out.size() != this->size()) {
		std::cout << "Error: Output vector size does not match input vector size\n";
		return -1;
	}
	const int block_size = 128;
	int num_blocks = (int)ceil(this->size() / block_size);
	_scalar_multiply << <num_blocks, block_size >> > (this->data_, scalar, this->size());
	cudaDeviceSynchronize();
	return 0;
}

int VectorND::vector_add(VectorND& out, VectorND& in)
{
	if (size_ != in.size()) {
		std::cout << "Error: Vector sizes do not match\n";
		return -1;
	}

	if (out.size() != size_) {
		std::cout << "Error: Output vector size does not match input vector size\n";
		return -1;
	}

	int block_size = 128;
	int num_blocks = (int)ceil(this->size() / block_size);
	if (num_blocks == 0) {
		num_blocks = 1;
	}
	_add_vector << <num_blocks, block_size >> > (out.data_, this->data_, in.data_, size_);
	cudaDeviceSynchronize();
	return 0;
}

int VectorND::vector_subtract(VectorND& out, VectorND& in)
{
	if (out.size() != in.size()) {
		return -1;
	}
	if (out.size() != this->size()) {
		return -1;
	}

	const int block_size = 128;
	int num_blocks = (int)ceil(this->size() / block_size);
	_subtract_vector << <num_blocks, block_size >> > (out.data_, this->data_, in.data_, this->size());
	cudaDeviceSynchronize();
	return 0;
}

int VectorND::vector_multiply(VectorND& out, VectorND& in)
{
	if (out.size() != in.size()) {
		return -1;
	}
	if (out.size() != this->size()) {
		return -1;
	}

	const int block_size = 128;
	int num_blocks = (int)ceil(this->size() / block_size);
	_multiply_vector << <num_blocks, block_size >> > (out.data_, this->data_, in.data_, this->size());
	cudaDeviceSynchronize();
	return 0;
}




