#include "CrossEntropy.cuh"

__global__ void cross_entropy_loss_kernel(float* out, float* prediction, float* truth, int rows, int columns) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (idx < columns) {
		float sum = 0;
		for (int i = 0; i < rows; i++) {	
			sum += truth[idx] * log(prediction[idx]);
		}
		atomicAdd(out, -sum);
	}


}

CrossEntropy::CrossEntropy()
{
}

CrossEntropy::~CrossEntropy()
{
}

float CrossEntropy::calculate_loss(Tensor2D& truth, Tensor2D& prediction) {
	float* d_out;

	const int num_threads = 128;
	int num_blocks = (int)ceil((float) truth.columns() / num_threads);
	if (num_blocks == 0) {
		num_blocks = 1;
	}
	cudaMalloc(&d_out, sizeof(float));
	cudaMemset(d_out, 0, sizeof(float));
	cross_entropy_loss_kernel << <num_blocks, target.get_columns() >> > (d_out, prediction.get_data(), truth.get_data(), target.get_rows(), target.get_columns());
	float out;
	cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	return out;
}

