#include "MSE.cuh"


__global__ void _mse_loss(float* out_err, float* target_data, float* predict_data, int rows, int columns) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (index < columns) {
		float sum = 0;
		for (int i = 0; i < rows; i++) {
			sum += pow((target_data[i * columns + index] - predict_data[i * columns + index]), 2);
		}
		sum /= columns;
		atomicAdd(out_err, sum);
	}
}	

__global__ void _mse_gradients(float* out_err, float* target_data, float* predict_data, int rows, int columns) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < rows * columns) {
		out_err[index] = 2 * (predict_data[index] - target_data[index]) / rows;
	}
}

float MSE::calculate_loss(Tensor2D& target, Tensor2D& prediction)
{
	if (target.rows() != prediction.rows() || target.columns() != prediction.columns())
	{
		std::cerr << "MSE::calculate_loss: input and output dimensions do not match" << std::endl;
		return -1;
	}
	float* d_loss; // loss variable on gpu
	cudaMalloc(&d_loss, sizeof(float));

	int block_size = 128;
	int num_blocks = ceil(((float)target.columns()) / block_size);
	if (num_blocks == 0) num_blocks = 1;
	_mse_loss << <num_blocks, block_size >> > (d_loss, target.data_, prediction.data_, target.rows(), target.columns());

	float loss;
	cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_loss);
	return loss;
}


int MSE::calculate_gradients(Tensor2D&target, Tensor2D& prediction, Tensor2D &out_err) {
	if (target.rows() != prediction.rows() || target.columns() != prediction.columns())
	{
		std::cerr << "MSE::calculate_gradients: input and output dimensions do not match" << std::endl;
		return -1;
	}
	if (out_err.rows() != target.rows() || out_err.columns() != target.columns())
	{
		std::cerr << "MSE::calculate_gradients: input and output dimensions do not match" << std::endl;
		return -1;
	}
	int size = target.rows() * target.columns();
	int block_size = 128;
	int num_blocks = ceil(((float)size) / block_size);
	if (num_blocks == 0) num_blocks = 1;
	_mse_gradients << <num_blocks, block_size >> > (out_err.data_, target.data_, prediction.data_, target.rows(), target.columns());
	cudaDeviceSynchronize();
	return 0;

}

MSE::~MSE() {
}

MSE::MSE() {
}
