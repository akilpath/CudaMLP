#include "Tensor2D.cuh"

#define TPB 256
#define BLOCK_DIM = 16

__global__ void _scalar_multiply(float* data, int max_idx, float scalar) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (index < max_idx) {
		data[index] = data[index] * scalar;
		index += stride;
	}
}

__global__ void _tensor_add(float* data_1, float* data_2, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < size) {
		data_1[x] += data_2[x];
	}
}

__global__ void _tensor_multiply(float* out, float* data_1, int r1, int c1, float* data_2, int r2, int c2) {
	__shared__ float data_s[18];

	float* data_1s = data_s;
	float* data_2s = data_s + 9;

	int output_rows = r1;
	int output_columns = c2;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float acc = 0;
	for (int k = 0; k < c1; k += blockDim.x) {
		if (threadIdx.x + k < c1 && y < r1) {
			data_1s[threadIdx.y*blockDim.x + threadIdx.x] = data_1[y * c1 + threadIdx.x + k];
		}
		if (x < c2 && threadIdx.y + k < r2) {
			data_2s[threadIdx.y * blockDim.x + threadIdx.x] = data_2[(threadIdx.y + k) * c2 + x];
		}
		__syncthreads();
		//if (x == 0 && y == 0) {
		//	printf("Data1s: %.2f, %.2f, %.2f\n", data_1s[threadIdx.y*blockDim.x], data_1s[threadIdx.y * blockDim.x + 1], data_1s[threadIdx.y * blockDim.x + 2]);
		//	printf("Data2s: %.2f, %.2f, %.2f\n", data_2s[threadIdx.x], data_2s[blockDim.x + threadIdx.x], data_2s[2 * blockDim.x + threadIdx.x]);
		//}
		if (x < output_columns && y < output_rows) {
			for (int i = 0; i < blockDim.x; i++) {
				if (i + k < c1) {
					//printf("deez");
					//if (x == 0 && y == 0) {
					//	printf("data1: %.2f, data2: %.2f\n", data_1s[threadIdx.y * blockDim.x + i], data_2s[(i * blockDim.x) + threadIdx.x]);
					//}
					//printf("data1: %.2f, data2: %.2f", data_1s[threadIdx.y * blockDim.x + i], data_2s[(i * blockDim.x) + threadIdx.x]);
					acc += data_1s[threadIdx.y*blockDim.x + i] * data_2s[(i*blockDim.x) + threadIdx.x];
				}
				//printf("acc y: %i, x: %i,  %i\n", y, x, acc);
			}
			out[y * output_columns + x] = acc;
		}
		//printf("acc y: %i, x: %i,  %i\n",y, x, acc);
		
	}
}


Tensor2D::Tensor2D(int rows, int columns){
	this->rows = rows;
	this->columns = columns;
	total_elements = rows * columns;
	cudaMallocManaged(&data, rows * columns * sizeof(float));
}

Tensor2D::~Tensor2D() {
	cudaFree(data);
}

void Tensor2D::set_value(int r, int c, float val){
	data[r * columns + c] = val;
}

float Tensor2D::get_value(int r, int c) {
	return data[r * columns + c];
}

void Tensor2D::print_data() {
	printf("[");
	for (int i = 0; i < rows; i++) {
		if (i == 0) {printf("[");}
		else {
			printf(" [");
		};

		
		for (int j = 0; j < columns; j++) {
			printf("%.3f, ", data[i * columns + j]);
		}
		if (i == rows - 1){
			printf("]");
		}
		else {
			printf("]\n");
		}
	}
	printf("]\n");
	std::cout << std::endl;
}

void Tensor2D::scalar_multiply(float scalar) {
	int num_blocks = (int) ceil(total_elements / 16.0);
	_scalar_multiply<<<num_blocks, 256>>>(data, total_elements, scalar);
	cudaDeviceSynchronize();
}

int Tensor2D::tensor_multiply(Tensor2D &out, Tensor2D &in) {
	if (this->columns != in.rows) {
		return -1;
	}
	if (out.rows != this->rows || out.columns != in.columns) {
		return -1;
	}
	dim3 block_size(3, 3);
	int num_blocks_y = (int) ceil(((float) out.rows) / 3);
	int num_blocks_x = (int) ceil(((float) out.columns) / 3);
	dim3 num_blocks(num_blocks_x, num_blocks_y);
	_tensor_multiply <<<num_blocks, block_size>>> (out.data, this->data, rows, columns, in.data, in.rows, in.columns);
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::tensor_add(Tensor2D& in) {
	if (this->columns != in.columns || this->rows != in.rows) {
		return -1;
	}

	int num_blocks = (int) ceil(this->total_elements / 128.0);
	_tensor_add << <num_blocks, 128 >> > (this->data, in.data, total_elements);
}

